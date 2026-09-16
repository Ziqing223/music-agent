"""P20 Fix07: same-batch sibling/duplicate suppression.

One recommendation batch must not present the same work twice -- the same
recording re-released under a different container (single / album versions) or
separate catalog identities that resolve to the same title and the same
canonical artist with nothing but the container differing. The observed UAT
shape: two distinct canonical tracks, verbatim-identical title, one canonical
artist, albums ``THE BOOK`` vs ``Ano yume wo nazotte - Single``.

Scope boundary (P20 Fix07 sec.7): this module selects which items appear in one
batch and exposes the same conservative identity for an explicitly supplied
historical-target filter. It never reads history itself, deletes catalog
entities, merges canonical entities, or modifies external identity, preference
state, or ``catalog_track_state``.

Identity priority (sec.4) -- deterministic, never model judgment:

1. same canonical target id            -> duplicate (defensive: the existing
   layers already deduplicate exact targets before ranking)
2. same ISRC, present on both items    -> duplicate
3. normalized title + identical canonical artist id set -> duplicate

Conservative bounds (sec.5/6):

* :func:`normalize_title` applies NFKC + casefold + whitespace collapse ONLY.
  It never deletes text, so explicit version markers (Live / Remix / Acoustic /
  English Version / Piano Version / Demo / Edit / feat. / with / ...) survive
  and distinct titles never merge.
* artist identity is the canonical artist id SET -- never name-string fuzz --
  so different artists are never deduped.
* missing identity FAILS OPEN: an item without a full (title, artist_ids)
  identity neither suppresses nor is suppressed by another item.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from music_agent.identity import (
    EntityType,
    IdentityValidationError,
    validate_canonical_id,
)
from music_agent.recommendation_contract import RecommendationItem

__all__ = [
    "TrackSiblingIdentity",
    "exclude_historical_siblings",
    "has_sibling_duplicate",
    "normalize_title",
    "resolve_sibling_identity",
    "select_distinct_works",
]


def normalize_title(title: str) -> str:
    """Conservative title normalization (sec.5).

    NFKC (Unicode normalization incl. common full/half-width equivalence),
    casefold, and collapse of every whitespace run to a single space. No text
    is deleted, so version markers never disappear.
    """
    if not isinstance(title, str):
        raise TypeError(f"title must be a str, got {type(title).__name__}")
    normalized = unicodedata.normalize("NFKC", title).casefold()
    return " ".join(normalized.split())


class TrackSiblingIdentity:
    """The deterministic identity surface used for sibling grouping (sec.4).

    ``artist_ids`` are canonical artist ids (never names). ``isrc`` is the
    recording ISRC from the track's external identity, ``None`` when absent.
    """

    __slots__ = ("title", "artist_ids", "isrc")

    def __init__(
        self, title: str, artist_ids: tuple[str, ...], isrc: str | None = None
    ) -> None:
        self.title = title
        self.artist_ids = artist_ids
        self.isrc = isrc

    def __repr__(self) -> str:
        return (
            f"TrackSiblingIdentity(title={self.title!r}, "
            f"artist_ids={self.artist_ids!r}, isrc={self.isrc!r})"
        )

    def group_keys(self) -> tuple[tuple[object, ...], ...]:
        """The single most authoritative group key this identity claims (sec.4).

        The ladder is sequential: when an ISRC is present it IS the recording
        identity -- the title+artist heuristic is not claimed at all, so a
        pair with two DIFFERENT ISRCs never merges (different recordings) and
        a pair where only one side carries an ISRC is undecidable and fails
        open. Without an ISRC the conservative normalized-title + canonical
        artist-set key is claimed instead.
        """
        if self.isrc:
            return (("isrc", self.isrc),)
        return (
            ("tt", normalize_title(self.title), frozenset(self.artist_ids)),
        )


def resolve_sibling_identity(track: Mapping[str, Any]) -> TrackSiblingIdentity | None:
    """Fail-open identity resolution from one model track.

    Returns ``None`` unless the full evidence is present: a non-empty string
    title, a non-empty sequence of string canonical artist ids, and (when
    present) a non-empty string ISRC inside ``external_ids``.
    """
    title = track.get("name")
    artist_ids = track.get("artist_ids")
    if not isinstance(title, str) or not title.strip():
        return None
    if not isinstance(artist_ids, (tuple, list)) or not artist_ids:
        return None
    ids = tuple(artist_id for artist_id in artist_ids if isinstance(artist_id, str))
    if len(ids) != len(artist_ids) or not ids:
        return None
    for artist_id in ids:
        try:
            validate_canonical_id(EntityType.ARTIST, artist_id)
        except IdentityValidationError:
            return None
    isrc: str | None = None
    external_ids = track.get("external_ids")
    if isinstance(external_ids, Mapping):
        candidate_isrc = external_ids.get("isrc")
        if isinstance(candidate_isrc, str) and candidate_isrc.strip():
            isrc = candidate_isrc
    return TrackSiblingIdentity(title=title, artist_ids=ids, isrc=isrc)


def _identity_for(
    item: RecommendationItem, track_by_id: Mapping[str, Mapping[str, Any]]
) -> TrackSiblingIdentity | None:
    track = track_by_id.get(item.candidate.target.target_id)
    if track is None:
        return None
    return resolve_sibling_identity(track)


def has_sibling_duplicate(
    items: Sequence[RecommendationItem],
    track_by_id: Mapping[str, Mapping[str, Any]],
) -> bool:
    """True when some later item shares a canonical target or any sibling key
    with an earlier item. Missing identities fail open (never duplicates)."""
    if len(items) < 2:
        return False
    seen_targets: set[str] = set()
    seen_keys: set[tuple[object, ...]] = set()
    for item in items:
        target_id = item.candidate.target.target_id
        if target_id in seen_targets:
            return True
        identity = _identity_for(item, track_by_id)
        keys = identity.group_keys() if identity is not None else ()
        if any(key in seen_keys for key in keys):
            return True
        seen_targets.add(target_id)
        seen_keys.update(keys)
    return False


def exclude_historical_siblings(
    items: Sequence[RecommendationItem],
    track_by_id: Mapping[str, Mapping[str, Any]],
    historical_target_ids: Sequence[str],
) -> tuple[RecommendationItem, ...]:
    """Drop exact or sibling matches to explicitly supplied historical targets.

    This is the cross-run counterpart to :func:`select_distinct_works`. It
    deliberately reuses :func:`resolve_sibling_identity` and its group-key
    ladder; no second title/artist/ISRC algorithm exists here. Missing historical
    metadata fails open for sibling matching while exact canonical IDs remain
    excluded. Item order is preserved and no same-batch selection is performed.
    """

    historical_ids = frozenset(historical_target_ids)
    historical_keys: set[tuple[object, ...]] = set()
    for target_id in historical_ids:
        track = track_by_id.get(target_id)
        identity = (
            resolve_sibling_identity(track) if track is not None else None
        )
        if identity is not None:
            historical_keys.update(identity.group_keys())

    selected: list[RecommendationItem] = []
    for item in items:
        target_id = item.candidate.target.target_id
        if target_id in historical_ids:
            continue
        identity = _identity_for(item, track_by_id)
        keys = identity.group_keys() if identity is not None else ()
        if any(key in historical_keys for key in keys):
            continue
        selected.append(item)
    return tuple(selected)


def select_distinct_works(
    items: Sequence[RecommendationItem],
    track_by_id: Mapping[str, Mapping[str, Any]],
    limit: int,
) -> tuple[RecommendationItem, ...]:
    """Deliver a same-batch selection with sibling suppression (sec.8/9).

    Walks ``items`` in their existing delivered order; an item is kept when
    neither its canonical target nor any of its sibling group keys matches an
    already-kept item, so the FIRST occurrence in the existing order wins and
    later siblings are skipped. Walking continues past skipped items (backfill
    from the caller-supplied pool) until ``limit`` items are kept or the pool
    is exhausted -- a shorter pool returns fewer items honestly (sec.10: no
    extra discovery is ever triggered for backfill).
    """
    if limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    kept: list[RecommendationItem] = []
    seen_targets: set[str] = set()
    seen_keys: set[tuple[object, ...]] = set()
    for item in items:
        target_id = item.candidate.target.target_id
        if target_id in seen_targets:
            continue
        identity = _identity_for(item, track_by_id)
        keys = identity.group_keys() if identity is not None else ()
        if any(key in seen_keys for key in keys):
            continue
        kept.append(item)
        seen_targets.add(target_id)
        seen_keys.update(keys)
        if len(kept) >= limit:
            break
    return tuple(kept)
