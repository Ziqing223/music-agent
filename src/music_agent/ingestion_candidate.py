"""In-memory ingestion candidates and a pure canonical Track promotion gate."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from music_agent.identity import (
    EXTERNAL_ID_FIXTURE_KEYS,
    EntityType,
    ExternalIdentityKey,
    IdentityValidationError,
    validate_canonical_id,
)
from music_agent.source_observation import ObservationState, ObservedValue
from music_agent.validation import (
    GraphValidationError,
    StructuralValidationError,
    validate_fixture,
    validate_structure,
)


class CandidateValidationError(ValueError):
    code = "validation_error"


class ArtistRelationState(StrEnum):
    UNRESOLVED = "unresolved"
    RESOLVED_TO_ARTISTS = "resolved_to_artists"


class AlbumRelationState(StrEnum):
    UNRESOLVED = "unresolved"
    RESOLVED_TO_ALBUM = "resolved_to_album"
    RESOLVED_ABSENT = "resolved_absent"


@dataclass(frozen=True, slots=True)
class ArtistRelationResolution:
    state: ArtistRelationState
    canonical_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.state, ArtistRelationState):
            raise CandidateValidationError("artist relation state is invalid")
        canonical_ids = tuple(self.canonical_ids)
        if self.state is ArtistRelationState.UNRESOLVED and canonical_ids:
            raise CandidateValidationError("unresolved artist relation cannot carry canonical IDs")
        object.__setattr__(self, "canonical_ids", canonical_ids)

    @classmethod
    def unresolved(cls) -> ArtistRelationResolution:
        return cls(ArtistRelationState.UNRESOLVED)

    @classmethod
    def resolved_to_artists(cls, canonical_ids: tuple[str, ...] | list[str]) -> ArtistRelationResolution:
        return cls(ArtistRelationState.RESOLVED_TO_ARTISTS, tuple(canonical_ids))


@dataclass(frozen=True, slots=True)
class AlbumRelationResolution:
    state: AlbumRelationState
    canonical_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, AlbumRelationState):
            raise CandidateValidationError("album relation state is invalid")
        if self.state is AlbumRelationState.RESOLVED_TO_ALBUM:
            if self.canonical_id is None:
                raise CandidateValidationError("resolved album relation requires a canonical ID")
        elif self.canonical_id is not None:
            raise CandidateValidationError(
                "unresolved or absent album relation cannot carry a canonical ID"
            )

    @classmethod
    def unresolved(cls) -> AlbumRelationResolution:
        return cls(AlbumRelationState.UNRESOLVED)

    @classmethod
    def resolved_to_album(cls, canonical_id: str) -> AlbumRelationResolution:
        return cls(AlbumRelationState.RESOLVED_TO_ALBUM, canonical_id)

    @classmethod
    def resolved_absent(cls) -> AlbumRelationResolution:
        return cls(AlbumRelationState.RESOLVED_ABSENT)


# Source systems the ingestion path accepts. ``apple_music`` is the Music.app persistent ID,
# ``apple_music_catalog`` the Apple Music Catalog Song ID, ``itunes_store`` the iTunes Store
# item ID. They are distinct namespaces: the scalar identity rule lets one canonical Track hold
# at most one ID of each kind, and none is ever inferred equal to another.
SUPPORTED_CANDIDATE_SOURCE_SYSTEMS = frozenset({"apple_music", "apple_music_catalog", "itunes_store"})


TRACK_SOURCE_FACT_PATHS = frozenset({
    "name",
    "duration_ms",
    "genres",
    "track_number",
    "disc_number",
    "release_date",
    "composer",
    "preview_url",
    "library_state.favorited",
    "library_state.disliked",
    "library_state.rating",
    "library_state.play_count",
    "library_state.skip_count",
    "library_state.added_to_library_at",
    "library_state.last_played_at",
})


@dataclass(frozen=True, slots=True)
class IngestionCandidate:
    external_identity: ExternalIdentityKey
    source_facts: Mapping[str, ObservedValue]
    artist_relation: ArtistRelationResolution = field(
        default_factory=ArtistRelationResolution.unresolved
    )
    album_relation: AlbumRelationResolution = field(
        default_factory=AlbumRelationResolution.unresolved
    )
    secondary_identities: tuple[ExternalIdentityKey, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.external_identity, ExternalIdentityKey):
            raise CandidateValidationError("external_identity must be an ExternalIdentityKey")
        if self.external_identity.source_system not in SUPPORTED_CANDIDATE_SOURCE_SYSTEMS:
            raise CandidateValidationError("unsupported source system")
        if self.external_identity.entity_type is not EntityType.TRACK:
            raise CandidateValidationError("P03.8.1 supports only Track candidates")
        secondaries = tuple(self.secondary_identities)
        if not all(isinstance(identity, ExternalIdentityKey) for identity in secondaries):
            raise CandidateValidationError("each secondary identity must be an ExternalIdentityKey")
        seen_keys: set[ExternalIdentityKey] = {self.external_identity}
        seen_sources: set[str] = {self.external_identity.source_system}
        for identity in secondaries:
            if identity.entity_type is not EntityType.TRACK:
                raise CandidateValidationError("secondary identities must be Track identities")
            if identity.source_system not in EXTERNAL_ID_FIXTURE_KEYS:
                raise CandidateValidationError(
                    f"secondary identity source system {identity.source_system!r} has no canonical fixture key"
                )
            if identity.source_system in seen_sources:
                # The scalar identity rule: one canonical Track holds at most one ID per system.
                raise CandidateValidationError(
                    f"duplicate {identity.source_system} identity on candidate"
                )
            if identity in seen_keys:
                raise CandidateValidationError("duplicate identity on candidate")
            seen_keys.add(identity)
            seen_sources.add(identity.source_system)
        object.__setattr__(self, "secondary_identities", secondaries)
        if not isinstance(self.source_facts, Mapping):
            raise CandidateValidationError("source_facts must be a mapping")
        facts: dict[str, ObservedValue] = {}
        for path, observed in self.source_facts.items():
            if path not in TRACK_SOURCE_FACT_PATHS:
                raise CandidateValidationError(f"unsupported candidate source fact: {path}")
            if not isinstance(observed, ObservedValue):
                raise CandidateValidationError(f"{path} must contain an ObservedValue")
            facts[path] = observed
        if not isinstance(self.artist_relation, ArtistRelationResolution):
            raise CandidateValidationError("artist_relation must be an ArtistRelationResolution")
        if not isinstance(self.album_relation, AlbumRelationResolution):
            raise CandidateValidationError("album_relation must be an AlbumRelationResolution")
        object.__setattr__(self, "source_facts", MappingProxyType(facts))


class PromotionBlockerCode(StrEnum):
    UNSUPPORTED_SOURCE = "unsupported_source"
    WRONG_ENTITY_TYPE = "wrong_entity_type"
    MISSING_EXTERNAL_ID = "missing_external_id"
    MISSING_NAME = "missing_name"
    ARTIST_RELATION_UNRESOLVED = "artist_relation_unresolved"
    ARTIST_RELATION_INVALID = "artist_relation_invalid"
    ALBUM_RELATION_UNRESOLVED = "album_relation_unresolved"
    ALBUM_RELATION_INVALID = "album_relation_invalid"
    GENRES_UNKNOWN = "genres_unknown"
    INVALID_SOURCE_FACT = "invalid_source_fact"


@dataclass(frozen=True, slots=True)
class PromotionBlocker:
    code: PromotionBlockerCode
    field_path: str
    detail: str


@dataclass(frozen=True, slots=True)
class PromotionEvaluation:
    blockers: tuple[PromotionBlocker, ...]

    @property
    def is_promotable(self) -> bool:
        return not self.blockers


class PromotionBlockedError(CandidateValidationError):
    code = "promotion_blocked"

    def __init__(self, evaluation: PromotionEvaluation) -> None:
        self.evaluation = evaluation
        codes = ", ".join(blocker.code.value for blocker in evaluation.blockers)
        super().__init__(f"candidate is not promotable: {codes}")


_PROBE_TRACK_ID = "trk_00000000-0000-4000-8000-000000000000"
_MISSING = ObservedValue.missing()


def evaluate_track_promotion(
    candidate: IngestionCandidate, canonical_model: dict[str, Any]
) -> PromotionEvaluation:
    if not isinstance(candidate, IngestionCandidate):
        raise CandidateValidationError("candidate must be an IngestionCandidate")
    try:
        validate_fixture(canonical_model)
    except (StructuralValidationError, GraphValidationError) as error:
        raise CandidateValidationError(f"current canonical model is invalid: {error}") from error

    blockers: list[PromotionBlocker] = []
    identity = candidate.external_identity
    if identity.source_system not in SUPPORTED_CANDIDATE_SOURCE_SYSTEMS:
        blockers.append(_blocker(
            PromotionBlockerCode.UNSUPPORTED_SOURCE, "external_identity.source_system",
            "only apple_music and apple_music_catalog are supported",
        ))
    if identity.entity_type is not EntityType.TRACK:
        blockers.append(_blocker(
            PromotionBlockerCode.WRONG_ENTITY_TYPE, "external_identity.entity_type",
            "only Track candidates are supported",
        ))
    if identity.external_id == "":
        blockers.append(_blocker(
            PromotionBlockerCode.MISSING_EXTERNAL_ID, "external_identity.external_id",
            "external ID must be non-empty",
        ))

    name = _fact(candidate, "name")
    name_is_missing = name.state in (ObservationState.MISSING, ObservationState.NULL) or (
        name.state is ObservationState.VALUE
        and isinstance(name.payload, str)
        and not name.payload.strip()
    )
    if name_is_missing:
        blockers.append(_blocker(
            PromotionBlockerCode.MISSING_NAME, "name", "a non-empty source name is required"
        ))

    genres = _fact(candidate, "genres")
    if genres.state is ObservationState.MISSING:
        blockers.append(_blocker(
            PromotionBlockerCode.GENRES_UNKNOWN, "genres",
            "MISSING genres cannot be initialized as a known-empty array",
        ))

    artist_ids: tuple[str, ...] = ()
    if candidate.artist_relation.state is ArtistRelationState.UNRESOLVED:
        blockers.append(_blocker(
            PromotionBlockerCode.ARTIST_RELATION_UNRESOLVED, "artist_ids",
            "artist relation must be reconciled before promotion",
        ))
    else:
        artist_ids = candidate.artist_relation.canonical_ids
        known_artists = {artist["id"] for artist in canonical_model["artists"]}
        if (
            not artist_ids
            or len(set(artist_ids)) != len(artist_ids)
            or not _all_valid_ids(EntityType.ARTIST, artist_ids)
            or any(artist_id not in known_artists for artist_id in artist_ids)
        ):
            blockers.append(_blocker(
                PromotionBlockerCode.ARTIST_RELATION_INVALID, "artist_ids",
                "resolved artists must be distinct existing canonical Artist IDs",
            ))

    album_id: str | None = None
    if candidate.album_relation.state is AlbumRelationState.UNRESOLVED:
        blockers.append(_blocker(
            PromotionBlockerCode.ALBUM_RELATION_UNRESOLVED, "album_id",
            "album relation must be reconciled before promotion",
        ))
    elif candidate.album_relation.state is AlbumRelationState.RESOLVED_TO_ALBUM:
        album_id = candidate.album_relation.canonical_id
        known_albums = {album["id"] for album in canonical_model["albums"]}
        if (
            album_id is None
            or not _is_valid_id(EntityType.ALBUM, album_id)
            or album_id not in known_albums
        ):
            blockers.append(_blocker(
                PromotionBlockerCode.ALBUM_RELATION_INVALID, "album_id",
                "resolved album must be an existing canonical Album ID",
            ))

    probe = _track_payload(
        candidate,
        _PROBE_TRACK_ID,
        (),
        None,
        name_override="validation probe" if name_is_missing else None,
        genres_override=[] if genres.state is ObservationState.MISSING else None,
    )
    try:
        validate_structure(_model_with_only_track(probe))
    except StructuralValidationError as error:
        blockers.append(_blocker(
            PromotionBlockerCode.INVALID_SOURCE_FACT,
            "source_facts",
            str(error),
        ))
    return PromotionEvaluation(tuple(blockers))


def build_promotable_track(
    candidate: IngestionCandidate,
    canonical_id: str,
    canonical_model: dict[str, Any],
) -> dict[str, Any]:
    evaluation = evaluate_track_promotion(candidate, canonical_model)
    if not evaluation.is_promotable:
        raise PromotionBlockedError(evaluation)
    try:
        validate_canonical_id(EntityType.TRACK, canonical_id)
    except IdentityValidationError as error:
        raise CandidateValidationError(str(error)) from error

    artist_ids = candidate.artist_relation.canonical_ids
    album_id = (
        candidate.album_relation.canonical_id
        if candidate.album_relation.state is AlbumRelationState.RESOLVED_TO_ALBUM
        else None
    )
    track = _track_payload(candidate, canonical_id, artist_ids, album_id)
    candidate_model = deepcopy(canonical_model)
    candidate_model["tracks"].append(deepcopy(track))
    try:
        validate_fixture(candidate_model)
    except (StructuralValidationError, GraphValidationError) as error:
        raise CandidateValidationError(f"constructed Track is invalid: {error}") from error
    return track


def _track_payload(
    candidate: IngestionCandidate,
    canonical_id: str,
    artist_ids: tuple[str, ...],
    album_id: str | None,
    *,
    name_override: str | None = None,
    genres_override: list[str] | None = None,
) -> dict[str, Any]:
    name = _canonical_value(_fact(candidate, "name")) if name_override is None else name_override
    genres = (
        _canonical_value(_fact(candidate, "genres"))
        if genres_override is None
        else genres_override
    )
    external_ids: dict[str, str | None] = {"apple_music_persistent_id": None}
    primary = candidate.external_identity
    external_ids[EXTERNAL_ID_FIXTURE_KEYS[primary.source_system]] = primary.external_id
    for secondary in candidate.secondary_identities:
        fixture_key = EXTERNAL_ID_FIXTURE_KEYS.get(secondary.source_system)
        if fixture_key is None:
            raise CandidateValidationError(
                f"secondary identity source system {secondary.source_system!r} has no canonical fixture key"
            )
        external_ids[fixture_key] = secondary.external_id
    return {
        "id": canonical_id,
        "external_ids": external_ids,
        "name": deepcopy(name),
        "artist_ids": list(artist_ids),
        "album_id": album_id,
        "duration_ms": _nullable_value(candidate, "duration_ms"),
        "genres": deepcopy(genres),
        "track_number": _nullable_value(candidate, "track_number"),
        "disc_number": _nullable_value(candidate, "disc_number"),
        "release_date": _nullable_value(candidate, "release_date"),
        "composer": _nullable_value(candidate, "composer"),
        "library_state": {
            "favorited": _nullable_value(candidate, "library_state.favorited"),
            "disliked": _nullable_value(candidate, "library_state.disliked"),
            "rating": _nullable_value(candidate, "library_state.rating"),
            "play_count": _nullable_value(candidate, "library_state.play_count"),
            "skip_count": _nullable_value(candidate, "library_state.skip_count"),
            "added_to_library_at": _nullable_value(
                candidate, "library_state.added_to_library_at"
            ),
            "last_played_at": _nullable_value(candidate, "library_state.last_played_at"),
        },
        "agent_metadata": {"tags": []},
    }


def _fact(candidate: IngestionCandidate, path: str) -> ObservedValue:
    return candidate.source_facts.get(path, _MISSING)


def _canonical_value(observed: ObservedValue) -> Any:
    if observed.state is ObservationState.VALUE:
        return deepcopy(observed.payload)
    return None


def _nullable_value(candidate: IngestionCandidate, path: str) -> Any:
    return _canonical_value(_fact(candidate, path))


def _all_valid_ids(entity_type: EntityType, canonical_ids: tuple[str, ...]) -> bool:
    return all(_is_valid_id(entity_type, canonical_id) for canonical_id in canonical_ids)


def _is_valid_id(entity_type: EntityType, canonical_id: object) -> bool:
    try:
        validate_canonical_id(entity_type, canonical_id)  # type: ignore[arg-type]
    except IdentityValidationError:
        return False
    return True


def _blocker(code: PromotionBlockerCode, field_path: str, detail: str) -> PromotionBlocker:
    return PromotionBlocker(code, field_path, detail)


def _model_with_only_track(track: dict[str, Any]) -> dict[str, Any]:
    return {
        "tracks": [track],
        "artists": [],
        "albums": [],
        "playlists": [],
        "playlist_memberships": [],
    }
