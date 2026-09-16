"""Pure, deterministic metadata similarity for canonical Tracks.

True Similarity V1 compares one already-resolved canonical seed Track with one
canonical candidate Track.  It performs no I/O, preference lookup, candidate
selection, persistence, playback, or Provider work.  Missing metadata contributes
zero and weights are never renormalized.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping

from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_propagation import canonicalize_genre_key
from music_agent.recommendation_contract import ScoreBreakdown, ScoreComponent


class TrackSimilarityError(ValueError):
    code = "track_similarity_error"


class TrackSimilarityValidationError(TrackSimilarityError):
    code = "validation_error"


GENRE_WEIGHT = 0.50
ARTIST_WEIGHT = 0.20
COMPOSER_WEIGHT = 0.10
TAG_WEIGHT = 0.10
DURATION_WEIGHT = 0.05
RELEASE_YEAR_WEIGHT = 0.05

SIMILARITY_SOURCE_PATH = "similarity_driven"

_RELEASE_YEAR = re.compile(r"^(\d{4})(?:-|$)")


@dataclass(frozen=True, slots=True)
class SimilarityExecutionContext:
    """Code-owned, turn-fixed authority for one current-track similarity seed."""

    seed_canonical_id: str

    def __post_init__(self) -> None:
        try:
            PreferenceTargetReference(
                PreferenceTargetKind.TRACK, self.seed_canonical_id
            )
        except ValueError as error:
            raise TrackSimilarityValidationError(
                "seed_canonical_id must be a canonical Track id"
            ) from error


@dataclass(frozen=True, slots=True)
class TrackSimilarityEvidence:
    """Complete V1 evidence and weighted score for one seed/candidate pair."""

    seed_id: str
    candidate_id: str
    shared_genres: tuple[str, ...]
    shared_artist_ids: tuple[str, ...]
    shared_composer: str | None
    shared_tags: tuple[str, ...]
    genre_overlap: float
    artist_overlap: float
    composer_overlap: float
    tag_overlap: float
    duration_proximity: float
    release_year_proximity: float
    categorical_eligible: bool
    total: float

    @property
    def score_breakdown(self) -> ScoreBreakdown:
        """Project the similarity-only score into the existing P07 score contract."""

        components = (
            ScoreComponent("seed_genre_overlap", GENRE_WEIGHT * self.genre_overlap),
            ScoreComponent("seed_artist_overlap", ARTIST_WEIGHT * self.artist_overlap),
            ScoreComponent(
                "seed_composer_overlap", COMPOSER_WEIGHT * self.composer_overlap
            ),
            ScoreComponent("seed_tag_overlap", TAG_WEIGHT * self.tag_overlap),
            ScoreComponent(
                "seed_duration_proximity",
                DURATION_WEIGHT * self.duration_proximity,
            ),
            ScoreComponent(
                "seed_release_year_proximity",
                RELEASE_YEAR_WEIGHT * self.release_year_proximity,
            ),
        )
        return ScoreBreakdown(self.total, components)


def score_track_similarity(
    seed: Mapping[str, Any], candidate: Mapping[str, Any]
) -> TrackSimilarityEvidence:
    """Return exact metadata evidence for ``candidate`` relative to ``seed``.

    Artist identity is canonical-ID equality.  Genre uses the project's frozen
    genre-key canonicalizer.  Composer and curated tags use normalized exact
    equality only; no token, alias, or fuzzy matching is performed.  Duration
    and release year are auxiliary and can never establish eligibility alone.
    """

    seed = _require_track(seed, "seed")
    candidate = _require_track(candidate, "candidate")
    seed_id = seed["id"]
    candidate_id = candidate["id"]

    seed_genres = _genre_set(seed.get("genres"), "seed.genres")
    candidate_genres = _genre_set(candidate.get("genres"), "candidate.genres")
    shared_genres = tuple(sorted(seed_genres & candidate_genres))
    genre_overlap = _jaccard(seed_genres, candidate_genres)

    seed_artists = _string_set(seed.get("artist_ids"), "seed.artist_ids")
    candidate_artists = _string_set(
        candidate.get("artist_ids"), "candidate.artist_ids"
    )
    shared_artists = tuple(sorted(seed_artists & candidate_artists))
    artist_overlap = _jaccard(seed_artists, candidate_artists)

    seed_composer = _normalized_optional_text(seed.get("composer"), "seed.composer")
    candidate_composer = _normalized_optional_text(
        candidate.get("composer"), "candidate.composer"
    )
    composer_overlap = float(
        seed_composer is not None and seed_composer == candidate_composer
    )
    shared_composer = (
        _display_text(seed.get("composer")) if composer_overlap else None
    )

    seed_tags = _tag_set(seed.get("agent_metadata"), "seed.agent_metadata")
    candidate_tags = _tag_set(
        candidate.get("agent_metadata"), "candidate.agent_metadata"
    )
    shared_tags = tuple(sorted(seed_tags & candidate_tags))
    tag_overlap = _jaccard(seed_tags, candidate_tags)

    duration_proximity = _duration_proximity(
        seed.get("duration_ms"), candidate.get("duration_ms")
    )
    release_year_proximity = _release_year_proximity(
        seed.get("release_date"), candidate.get("release_date")
    )
    categorical_eligible = any(
        value > 0.0
        for value in (genre_overlap, artist_overlap, composer_overlap, tag_overlap)
    )
    total = sum(
        (
            GENRE_WEIGHT * genre_overlap,
            ARTIST_WEIGHT * artist_overlap,
            COMPOSER_WEIGHT * composer_overlap,
            TAG_WEIGHT * tag_overlap,
            DURATION_WEIGHT * duration_proximity,
            RELEASE_YEAR_WEIGHT * release_year_proximity,
        )
    )
    return TrackSimilarityEvidence(
        seed_id=seed_id,
        candidate_id=candidate_id,
        shared_genres=shared_genres,
        shared_artist_ids=shared_artists,
        shared_composer=shared_composer,
        shared_tags=shared_tags,
        genre_overlap=genre_overlap,
        artist_overlap=artist_overlap,
        composer_overlap=composer_overlap,
        tag_overlap=tag_overlap,
        duration_proximity=duration_proximity,
        release_year_proximity=release_year_proximity,
        categorical_eligible=categorical_eligible,
        total=total,
    )


def _require_track(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrackSimilarityValidationError(f"{label} must be a Track mapping")
    track_id = value.get("id")
    try:
        PreferenceTargetReference(PreferenceTargetKind.TRACK, track_id)
    except (TypeError, ValueError) as error:
        raise TrackSimilarityValidationError(
            f"{label}.id must be a canonical Track id"
        ) from error
    return value


def _string_set(value: object, label: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, (list, tuple)):
        raise TrackSimilarityValidationError(f"{label} must be a sequence")
    if any(not isinstance(entry, str) for entry in value):
        raise TrackSimilarityValidationError(f"{label} must contain strings")
    return frozenset(entry for entry in value if entry)


def _genre_set(value: object, label: str) -> frozenset[str]:
    return frozenset(
        key
        for key in (
            canonicalize_genre_key(entry) for entry in _string_set(value, label)
        )
        if key
    )


def _normalized_optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TrackSimilarityValidationError(f"{label} must be a string or None")
    normalized = unicodedata.normalize("NFC", " ".join(value.split())).casefold()
    return normalized or None


def _display_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value or None


def _tag_set(value: object, label: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, Mapping):
        raise TrackSimilarityValidationError(f"{label} must be a mapping")
    tags = _string_set(value.get("tags"), f"{label}.tags")
    return frozenset(
        normalized
        for normalized in (
            _normalized_optional_text(tag, f"{label}.tags") for tag in tags
        )
        if normalized is not None
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _duration_proximity(seed_ms: object, candidate_ms: object) -> float:
    for value, label in ((seed_ms, "seed.duration_ms"), (candidate_ms, "candidate.duration_ms")):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise TrackSimilarityValidationError(
                f"{label} must be a non-negative integer or None"
            )
    if not seed_ms or not candidate_ms:
        return 0.0
    delta_seconds = abs(seed_ms - candidate_ms) / 1000.0
    return max(0.0, 1.0 - delta_seconds / 180.0)


def _release_year(value: object, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TrackSimilarityValidationError(f"{label} must be a string or None")
    match = _RELEASE_YEAR.match(value)
    if match is None:
        raise TrackSimilarityValidationError(f"{label} has no canonical release year")
    return int(match.group(1))


def _release_year_proximity(seed_date: object, candidate_date: object) -> float:
    seed_year = _release_year(seed_date, "seed.release_date")
    candidate_year = _release_year(candidate_date, "candidate.release_date")
    if seed_year is None or candidate_year is None:
        return 0.0
    return max(0.0, 1.0 - abs(seed_year - candidate_year) / 10.0)
