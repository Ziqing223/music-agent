"""P10.17b: Signed genre-affinity reducer (P10-owned derived layer).

Consumes source-tagged GENRE contributions produced by the sealed
``propagate_track_preference`` and reduces them into bounded, deterministic,
source-scoped genre affinities. Frozen policy per the phase-owner approval:

- aggregation: pos_sum / neg_sum arithmetic; net = pos_sum - neg_sum;
  affinity = clamp(net / saturation_divisor, -1, +1);
- saturation_divisor and affinity_threshold are INJECTED unfrozen policy
  seams (the saturation calibration the sealed propagation docstring
  explicitly reserves);
- minimum evidence: at least one contribution and |affinity| >= threshold;
- one explicit dislike lowers the genre aggregate by exactly its attenuated
  contribution magnitude divided by the divisor (bounded, never amplified);
- aggregates are keyed by (source_system, canonical genre key) -- equal
  genre strings from different sources never mix;
- track-level inference picks the strongest |affinity| genre (sign
  preserved; tie-break by key order) and only applies to targets whose
  direct state is non-directional (the caller enforces that boundary).

Pure and deterministic: no clock, no store, no randomness, no persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from music_agent.preference_attribution import PreferenceTargetKind, PreferenceTargetReference
from music_agent.preference_propagation import (
    InferredAffinityContribution,
    PropagationKind,
    canonicalize_genre_key,
)
from music_agent.preference_strength import PreferenceState, PreferenceStrength


class GenreAffinityError(ValueError):
    code = "genre_affinity_error"


@dataclass(frozen=True, slots=True)
class GenreAffinityPolicy:
    """The unfrozen calibration seams of the genre-affinity reducer."""

    saturation_divisor: float = 1.0
    affinity_threshold: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.saturation_divisor, (int, float)) or self.saturation_divisor <= 0:
            raise GenreAffinityError("saturation_divisor must be positive")
        if not isinstance(self.affinity_threshold, (int, float)):
            raise GenreAffinityError("affinity_threshold must be a number")
        if not 0 < self.affinity_threshold <= 1:
            raise GenreAffinityError("affinity_threshold must be within (0, 1]")


@dataclass(frozen=True, slots=True)
class SourcedContribution:
    """One genre contribution tagged with the source of its direct preference."""

    source_system: str
    contribution: InferredAffinityContribution

    def __post_init__(self) -> None:
        if not isinstance(self.source_system, str) or self.source_system == "":
            raise GenreAffinityError("source_system must be a non-empty string")
        if not isinstance(self.contribution, InferredAffinityContribution):
            raise GenreAffinityError("contribution must be an InferredAffinityContribution")
        if self.contribution.kind is not PropagationKind.GENRE:
            raise GenreAffinityError("only GENRE contributions feed the genre reducer")
        if self.contribution.target.kind is not PreferenceTargetKind.GENRE:
            raise GenreAffinityError("contribution target must be a GENRE")


@dataclass(frozen=True, slots=True)
class GenreAffinity:
    """One reduced (source, genre) affinity with its full contribution set."""

    source_system: str
    genre_key: str
    positive_count: int
    negative_count: int
    positive_sum: float
    negative_sum: float
    net: float
    affinity: float
    source_track_ids: tuple[str, ...]

    def explanation(self) -> str:
        return (
            f"推断（genre: {self.genre_key}；来源: {self.source_system}；"
            f"正面 {self.positive_count} 首 / 负面 {self.negative_count} 首；"
            f"净支持 {round(self.net, 3)}）"
        )


def build_genre_affinities(
    sourced: Iterable[SourcedContribution],
    policy: GenreAffinityPolicy | None = None,
) -> tuple[GenreAffinity, ...]:
    """Reduce source-tagged genre contributions into (source, genre) affinities.

    Deterministic: grouped by (source_system, canonical genre key), keys sorted.
    Contributions whose canonicalized genre key is empty are rejected (the sealed
    propagation rule) and fail closed.
    """
    policy = policy or GenreAffinityPolicy()
    if not isinstance(policy, GenreAffinityPolicy):
        raise GenreAffinityError("policy must be a GenreAffinityPolicy")
    buckets: dict[tuple[str, str], dict] = {}
    for sourced_item in sourced:
        if not isinstance(sourced_item, SourcedContribution):
            raise GenreAffinityError("each item must be a SourcedContribution")
        contribution = sourced_item.contribution
        genre_key = canonicalize_genre_key(contribution.target.target_id)
        if not genre_key:
            raise GenreAffinityError("genre key canonicalizes to an empty string")
        key = (sourced_item.source_system, genre_key)
        bucket = buckets.setdefault(
            key,
            {
                "positive_count": 0,
                "negative_count": 0,
                "positive_sum": 0.0,
                "negative_sum": 0.0,
                "tracks": [],
            },
        )
        if contribution.direction.value == "positive":
            bucket["positive_count"] += 1
            bucket["positive_sum"] += contribution.derived_magnitude
        else:  # negative
            bucket["negative_count"] += 1
            bucket["negative_sum"] += contribution.derived_magnitude
        if contribution.source_track.target_id not in bucket["tracks"]:
            bucket["tracks"].append(contribution.source_track.target_id)
    affinities: list[GenreAffinity] = []
    for key in sorted(buckets):
        bucket = buckets[key]
        source_system, genre_key = key
        net = round(bucket["positive_sum"] - bucket["negative_sum"], 6)
        affinity = round(max(-1.0, min(1.0, net / policy.saturation_divisor)), 6)
        affinities.append(
            GenreAffinity(
                source_system=source_system,
                genre_key=genre_key,
                positive_count=bucket["positive_count"],
                negative_count=bucket["negative_count"],
                positive_sum=bucket["positive_sum"],
                negative_sum=bucket["negative_sum"],
                net=net,
                affinity=affinity,
                source_track_ids=tuple(bucket["tracks"]),
            )
        )
    return tuple(affinities)


def select_track_affinity(
    genres: Iterable[str],
    source_system: str,
    affinities: Iterable[GenreAffinity],
    policy: GenreAffinityPolicy | None = None,
) -> GenreAffinity | None:
    """Pick the strongest |affinity| genre for one track (sign preserved).

    Only affinities of the given source are considered; equal genre strings from
    other sources are invisible here. Tie-break: genre key ascending. An affinity
    below the threshold is treated as absent (insufficient evidence).
    """
    policy = policy or GenreAffinityPolicy()
    if not isinstance(policy, GenreAffinityPolicy):
        raise GenreAffinityError("policy must be a GenreAffinityPolicy")
    by_key = {
        affinity.genre_key: affinity
        for affinity in affinities
        if isinstance(affinity, GenreAffinity) and affinity.source_system == source_system
    }
    best: GenreAffinity | None = None
    for genre in genres:
        key = canonicalize_genre_key(genre)
        if not key or key not in by_key:
            continue
        candidate = by_key[key]
        if abs(candidate.affinity) < policy.affinity_threshold:
            continue
        if best is None or abs(candidate.affinity) > abs(best.affinity) or (
            abs(candidate.affinity) == abs(best.affinity) and key < best.genre_key
        ):
            best = candidate
    return best


def infer_track_affinity(
    track_id: str,
    genres: Iterable[str],
    source_system: str,
    affinities: Iterable[GenreAffinity],
    policy: GenreAffinityPolicy | None = None,
) -> PreferenceStrength | None:
    """One track-level inferred strength from its genres' source-scoped affinities.

    Returns None when no genre clears the threshold (insufficient). The sign is
    the selected genre's net sign; the magnitude is the bounded |affinity|.
    """
    selected = select_track_affinity(genres, source_system, affinities, policy)
    if selected is None or selected.affinity == 0:
        return None
    state = PreferenceState.POSITIVE if selected.affinity > 0 else PreferenceState.NEGATIVE
    return PreferenceStrength(state, round(abs(selected.affinity), 4))
