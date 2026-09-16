"""Stable canonical ID generation and in-memory external identity binding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID, uuid4


class EntityType(StrEnum):
    TRACK = "track"
    ARTIST = "artist"
    ALBUM = "album"
    PLAYLIST = "playlist"
    PLAYLIST_MEMBERSHIP = "playlist_membership"


ENTITY_ID_PREFIX: dict[EntityType, str] = {
    EntityType.TRACK: "trk_",
    EntityType.ARTIST: "art_",
    EntityType.ALBUM: "alb_",
    EntityType.PLAYLIST: "pl_",
    EntityType.PLAYLIST_MEMBERSHIP: "pm_",
}


# Source systems whose external IDs project into the canonical fixture's ``external_ids`` object.
# ``apple_music`` is the Music.app persistent ID, ``apple_music_catalog`` the Apple Music Catalog
# Song ID, ``isrc`` the recording's ISRC, and ``itunes_store`` the iTunes Store item ID (never
# assumed equal to a Catalog ID). All four may identify one canonical entity.
EXTERNAL_ID_FIXTURE_KEYS: dict[str, str] = {
    "apple_music": "apple_music_persistent_id",
    "apple_music_catalog": "apple_music_catalog_id",
    "isrc": "isrc",
    "itunes_store": "itunes_store_id",
}

# Source systems under which one canonical entity holds at most one external ID. Enforced both in
# code (repository) and by partial unique indexes (migrations 0001 and 0017).
SCALAR_SOURCE_SYSTEMS = frozenset(EXTERNAL_ID_FIXTURE_KEYS)


class IdentityError(ValueError):
    code = "identity_error"


class IdentityValidationError(IdentityError):
    code = "validation_error"


class IdentityConflictError(IdentityError):
    code = "identity_conflict"

    def __init__(
        self,
        key: ExternalIdentityKey,
        existing_canonical_id: str,
        attempted_canonical_id: str,
    ) -> None:
        self.key = key
        self.existing_canonical_id = existing_canonical_id
        self.attempted_canonical_id = attempted_canonical_id
        super().__init__(
            f"External identity {key!r} is already bound to {existing_canonical_id}; "
            f"cannot bind it to {attempted_canonical_id}"
        )


@dataclass(frozen=True, slots=True)
class ExternalIdentityKey:
    source_system: str
    entity_type: EntityType
    external_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_system, str) or self.source_system == "":
            raise IdentityValidationError("source_system must be a non-empty string")
        _require_entity_type(self.entity_type)
        if not isinstance(self.external_id, str) or self.external_id == "":
            raise IdentityValidationError("external_id must be a non-empty string")


def generate_canonical_id(entity_type: EntityType) -> str:
    """Generate a new entity-prefixed canonical ID with a random UUIDv4 suffix."""
    entity_type = _require_entity_type(entity_type)
    return f"{ENTITY_ID_PREFIX[entity_type]}{uuid4()}"


def validate_canonical_id(entity_type: EntityType, canonical_id: str) -> None:
    """Validate canonical shape and namespace without requiring a UUIDv4 suffix."""
    entity_type = _require_entity_type(entity_type)
    if not isinstance(canonical_id, str):
        raise IdentityValidationError("canonical_id must be a string")
    prefix = ENTITY_ID_PREFIX[entity_type]
    if not canonical_id.startswith(prefix):
        raise IdentityValidationError(
            f"canonical_id for {entity_type.value} must use the {prefix} namespace"
        )
    suffix = canonical_id[len(prefix) :]
    try:
        parsed = UUID(suffix)
    except (AttributeError, ValueError) as error:
        raise IdentityValidationError("canonical_id suffix must be a canonical UUID") from error
    if str(parsed) != suffix or parsed.version not in {1, 2, 3, 4, 5}:
        raise IdentityValidationError("canonical_id suffix must be a canonical UUID")


def _require_entity_type(entity_type: object) -> EntityType:
    if not isinstance(entity_type, EntityType):
        raise IdentityValidationError("entity_type must be an EntityType")
    return entity_type


class IdentityRegistry:
    """Non-persistent index from external identity keys to canonical IDs."""

    def __init__(self) -> None:
        self._bindings: dict[ExternalIdentityKey, str] = {}

    def create_canonical_id(self, entity_type: EntityType) -> str:
        return generate_canonical_id(entity_type)

    def bind(self, key: ExternalIdentityKey, canonical_id: str) -> str:
        if not isinstance(key, ExternalIdentityKey):
            raise IdentityValidationError("key must be an ExternalIdentityKey")
        validate_canonical_id(key.entity_type, canonical_id)
        existing = self._bindings.get(key)
        if existing is None:
            self._bindings[key] = canonical_id
            return canonical_id
        if existing == canonical_id:
            return canonical_id
        raise IdentityConflictError(key, existing, canonical_id)

    def lookup(self, key: ExternalIdentityKey) -> str | None:
        if not isinstance(key, ExternalIdentityKey):
            raise IdentityValidationError("key must be an ExternalIdentityKey")
        return self._bindings.get(key)

    def __len__(self) -> int:
        return len(self._bindings)
