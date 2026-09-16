"""P11.2: Signed artist-affinity reducer (the artist twin of the P10.17b genre reducer).

Consumes source-tagged ARTIST contributions produced by the sealed
``propagate_track_preference`` and reduces them into bounded, deterministic,
source-scoped artist affinities. The reducer math and the calibration seams are
identical to the genre reducer (:class:`~music_agent.genre_affinity.GenreAffinityPolicy`
is reused unchanged -- one calibration governs both dimensions), with one deliberate
difference: artist targets are canonical Artist IDs, so no key canonicalization is
applied -- the canonical ID *is* the key.

Pure and deterministic: no clock, no store, no randomness, no persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from music_agent.genre_affinity import GenreAffinityPolicy
from music_agent.preference_attribution import PreferenceTargetKind
from music_agent.preference_propagation import (
    InferredAffinityContribution,
    PropagationKind,
)


class ArtistAffinityError(ValueError):
    code = "artist_affinity_error"


@dataclass(frozen=True, slots=True)
class SourcedArtistContribution:
    """One artist contribution tagged with the source of its direct preference."""

    source_system: str
    contribution: InferredAffinityContribution

    def __post_init__(self) -> None:
        if not isinstance(self.source_system, str) or self.source_system == "":
            raise ArtistAffinityError("source_system must be a non-empty string")
        if not isinstance(self.contribution, InferredAffinityContribution):
            raise ArtistAffinityError("contribution must be an InferredAffinityContribution")
        if self.contribution.kind is not PropagationKind.ARTIST:
            raise ArtistAffinityError("only ARTIST contributions feed the artist reducer")
        if self.contribution.target.kind is not PreferenceTargetKind.ARTIST:
            raise ArtistAffinityError("contribution target must be an ARTIST")


@dataclass(frozen=True, slots=True)
class ArtistAffinity:
    """One reduced (source, canonical artist) affinity with its full contribution set."""

    source_system: str
    artist_id: str
    positive_count: int
    negative_count: int
    positive_sum: float
    negative_sum: float
    net: float
    affinity: float
    source_track_ids: tuple[str, ...]

    def explanation(self) -> str:
        return (
            f"推断（artist: {self.artist_id}；来源: {self.source_system}；"
            f"正面 {self.positive_count} 首 / 负面 {self.negative_count} 首；"
            f"净支持 {round(self.net, 3)}）"
        )


def build_artist_affinities(
    sourced: Iterable[SourcedArtistContribution],
    policy: GenreAffinityPolicy | None = None,
) -> tuple[ArtistAffinity, ...]:
    """Reduce source-tagged artist contributions into (source, canonical artist) affinities.

    Deterministic: grouped by (source_system, canonical artist ID), keys sorted. The
    reducer math is the same as the genre reducer: pos_sum / neg_sum arithmetic,
    net = pos_sum - neg_sum, affinity = clamp(net / saturation_divisor, -1, +1).
    """
    policy = policy or GenreAffinityPolicy()
    if not isinstance(policy, GenreAffinityPolicy):
        raise ArtistAffinityError("policy must be a GenreAffinityPolicy")
    buckets: dict[tuple[str, str], dict] = {}
    for sourced_item in sourced:
        if not isinstance(sourced_item, SourcedArtistContribution):
            raise ArtistAffinityError("each item must be a SourcedArtistContribution")
        contribution = sourced_item.contribution
        key = (sourced_item.source_system, contribution.target.target_id)
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
        else:
            bucket["negative_count"] += 1
            bucket["negative_sum"] += contribution.derived_magnitude
        if contribution.source_track.target_id not in bucket["tracks"]:
            bucket["tracks"].append(contribution.source_track.target_id)
    affinities: list[ArtistAffinity] = []
    for key in sorted(buckets):
        bucket = buckets[key]
        source_system, artist_id = key
        net = round(bucket["positive_sum"] - bucket["negative_sum"], 6)
        affinity = round(max(-1.0, min(1.0, net / policy.saturation_divisor)), 6)
        affinities.append(
            ArtistAffinity(
                source_system=source_system,
                artist_id=artist_id,
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
