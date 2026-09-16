"""P15-S3-S1: durable long-term state per canonical Catalog track.

``catalog_track_state`` (migration 0019) is derived, re-buildable memory over the authoritative
canonical store and recommendation history -- one row per canonical Catalog track. This module is
the single write/read surface for that state during P15-S3:

- ``ensure_state`` creates the row for a canonical Track when none exists (zeroed discovery and
  recommendation facts -- "0" means no durable historical events recorded, never "never
  happened").
- ``record_discovery_occurrence`` is the discovery authoritative write hook: one real catalog
  discovery occurrence (fresh promotion or ALREADY_BOUND hit) bumps ``discovery_count``, stamps
  ``last_discovered_at``, sets ``first_discovered_at`` exactly once, and merges the normalized
  search term into a bounded summary. Callers are the catalog ingestion orchestrator only --
  never the provider agent layer, never tool-call traces.
- ``get_state`` is the minimal read for S1 and later slices.

The recommendation projection columns (``first_recommended_at`` / ``last_recommended_at`` /
``recommendation_count``) are re-buildable at any time from ``recommendation_runs``: migration
0019 backfills them historically, and the two runtime ``save_result`` write points project each
new persisted run through ``record_recommendation_items`` (P15-S3-S2) over the same item口径.
``find_states_by_term`` is a read-only memory view only -- term reuse / freshness policy over
live catalog search is P15-S3-S3 territory and never implemented here. No exploration score,
queue position, decay, eligibility, cached score, or preference state lives in this table.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from music_agent.identity import EntityType, validate_canonical_id
from music_agent.repository import _open_store_connection

# Bounded term summary constants: the discovery-term memory is a small recent-sample map, not a
# provenance event log. Terms are the user-visible search terms already flowing through the
# catalog tool; the map caps both the number of distinct terms and each term's stored length.
MAX_DISCOVERY_TERMS = 8
MAX_DISCOVERY_TERM_LENGTH = 200

# Row-count chunk for ``load_states`` IN clauses (P15-S3-S3A): stays far below the
# SQLite bind-variable limit even for a full catalog pool read.
_LOAD_STATES_CHUNK = 250

_CATALOG_SOURCE_SYSTEMS = ("itunes_store", "apple_music_catalog")


class CatalogTrackStateError(ValueError):
    code = "catalog_track_state_error"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_discovery_term(term: object) -> str | None:
    """Normalize one catalog search term for the bounded memory: casefolded, stripped.

    Empty or whitespace-only input yields ``None`` (nothing worth remembering); the stored text
    is truncated to ``MAX_DISCOVERY_TERM_LENGTH``.
    """
    if not isinstance(term, str):
        return None
    normalized = term.strip().casefold()
    if normalized == "":
        return None
    return normalized[:MAX_DISCOVERY_TERM_LENGTH]


def merge_discovery_terms(
    existing: dict[str, int] | None, term: str | None
) -> dict[str, int]:
    """Merge one normalized term into the bounded term summary.

    Repeated terms accumulate their count and move to the newest position; a brand-new term is
    appended and, once ``MAX_DISCOVERY_TERMS`` distinct terms are held, the oldest-inserted term
    is evicted (FIFO). ``None``/missing terms merge to the unchanged map.
    """
    merged: dict[str, int] = dict(existing or {})
    normalized = normalize_discovery_term(term)
    if normalized is None:
        return merged
    merged.pop(normalized, None)
    merged[normalized] = (existing or {}).get(normalized, 0) + 1
    while len(merged) > MAX_DISCOVERY_TERMS:
        merged.pop(next(iter(merged)))
    return merged


def _encode_terms(terms: dict[str, int]) -> str:
    return json.dumps(terms, ensure_ascii=False, separators=(",", ":"))


def _decode_terms(text: object) -> dict[str, int]:
    if text is None:
        return {}
    if not isinstance(text, str):
        raise CatalogTrackStateError("discovery_terms_json must be text")
    try:
        decoded = json.loads(text)
    except ValueError as error:
        raise CatalogTrackStateError(
            f"discovery_terms_json is not valid JSON: {error}"
        ) from error
    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) and isinstance(value, int) and value > 0
        for key, value in decoded.items()
    ):
        raise CatalogTrackStateError(
            "discovery_terms_json must be a {term: positive-count} object"
        )
    return decoded


def _parse_instant(text: str) -> datetime:
    """Parse a stored recommendation instant back to an aware datetime (fail closed).

    Recommendation instants are always timezone-aware ISO 8601 text (``produced_at``
    mirrors of recommendation history); anything else is corruption and fails loudly
    instead of being silently mis-ordered.
    """
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise CatalogTrackStateError(
            f"stored recommendation instant {text!r} is not ISO 8601"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CatalogTrackStateError(
            f"stored recommendation instant {text!r} is not timezone-aware"
        )
    return parsed


def _earlier_by_backfill_order(
    candidate: datetime, candidate_text: str, stored_text: str
) -> bool:
    """Migration-B ordering: candidate is earlier than the stored instant by true instant,
    with verbatim lexical text as the same-instant tie-break (B's ``produced_at`` secondary)."""
    stored = _parse_instant(stored_text)
    if candidate == stored:
        return candidate_text < stored_text
    return candidate < stored


def _later_by_backfill_order(
    candidate: datetime, candidate_text: str, stored_text: str
) -> bool:
    """Symmetric to ``_earlier_by_backfill_order`` for the newest instant."""
    stored = _parse_instant(stored_text)
    if candidate == stored:
        return candidate_text > stored_text
    return candidate > stored


@dataclass(frozen=True, slots=True)
class CatalogTrackState:
    """One canonical Catalog track's long-term memory row."""

    canonical_id: str
    source_system: str
    first_discovered_at: str | None
    last_discovered_at: str | None
    discovery_count: int
    discovery_terms: dict[str, int]
    first_recommended_at: str | None
    last_recommended_at: str | None
    recommendation_count: int
    updated_at: str


def _decode_state_row(row: sqlite3.Row) -> CatalogTrackState:
    """Decode one ``catalog_track_state`` row into its frozen domain value."""
    return CatalogTrackState(
        canonical_id=row[0],
        source_system=row[1],
        first_discovered_at=row[2],
        last_discovered_at=row[3],
        discovery_count=int(row[4]),
        discovery_terms=_decode_terms(row[5]),
        first_recommended_at=row[6],
        last_recommended_at=row[7],
        recommendation_count=int(row[8]),
        updated_at=row[9],
    )


class CatalogTrackStateRepository:
    """Write/read the derived ``catalog_track_state`` rows inside the shared SQLite store."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._connection = _open_store_connection(self.database_path)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> CatalogTrackStateRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()
        return int(row[0])

    def ensure_state(self, canonical_id: str, *, source_system: str) -> bool:
        """Create the zeroed row for ``canonical_id`` when none exists.

        Returns ``True`` when a new row was created, ``False`` when one already existed. All
        discovery and recommendation facts start at the honest empty values: NULL instants,
        counts of 0, no terms. ``source_system`` is the track's catalog origin system.
        """
        canonical_id = _require_canonical_id(canonical_id)
        source_system = _require_source_system(source_system)
        existing = self._connection.execute(
            "SELECT 1 FROM catalog_track_state WHERE canonical_id=?",
            (canonical_id,),
        ).fetchone()
        if existing is not None:
            return False
        # Plain INSERT (no OR IGNORE): a foreign-key or CHECK violation must stay loud, never
        # silently turn into a no-op.
        self._connection.execute(
            """INSERT INTO catalog_track_state(
                canonical_id, entity_type, source_system
            ) VALUES (?, 'track', ?)""",
            (canonical_id, source_system),
        )
        return True

    def record_discovery_occurrence(
        self,
        canonical_id: str,
        *,
        source_system: str,
        term: str | None = None,
        now: datetime | None = None,
    ) -> None:
        """Record one real catalog discovery occurrence for ``canonical_id``.

        ``now`` is the injected reference instant (aware UTC datetime; the system clock is only
        the default, so tests inject deterministic values). On first occurrence the row is
        created with ``first_discovered_at == last_discovered_at`` and ``discovery_count == 1``;
        later occurrences bump the count and move ``last_discovered_at`` while
        ``first_discovered_at`` stays immutable. The normalized ``term`` (when present) is
        merged into the bounded term summary. Recommendation projection columns are never
        touched here (S3-S2 wires the runtime ``save_result`` hook).
        """
        canonical_id = _require_canonical_id(canonical_id)
        source_system = _require_source_system(source_system)
        stamped = now if now is not None else _utc_now()
        if not isinstance(stamped, datetime) or stamped.tzinfo is None:
            raise CatalogTrackStateError("now must be an aware datetime")
        stamp = stamped.isoformat()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            inserted = self.ensure_state(canonical_id, source_system=source_system)
            if inserted:
                terms = merge_discovery_terms({}, term)
                self._connection.execute(
                    """UPDATE catalog_track_state SET
                        first_discovered_at=?, last_discovered_at=?,
                        discovery_count=1, discovery_terms_json=?, updated_at=?
                    WHERE canonical_id=?""",
                    (stamp, stamp, _encode_terms(terms), stamp, canonical_id),
                )
            else:
                row = self._connection.execute(
                    """SELECT discovery_count, discovery_terms_json
                    FROM catalog_track_state WHERE canonical_id=?""",
                    (canonical_id,),
                ).fetchone()
                if row is None:
                    raise CatalogTrackStateError(
                        f"state row for {canonical_id!r} disappeared during recording"
                    )
                terms = merge_discovery_terms(_decode_terms(row[1]), term)
                self._connection.execute(
                    """UPDATE catalog_track_state SET
                        last_discovered_at=?, discovery_count=discovery_count+1,
                        discovery_terms_json=?, updated_at=?
                    WHERE canonical_id=?""",
                    (stamp, _encode_terms(terms), stamp, canonical_id),
                )
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def record_recommendation_items(
        self,
        track_ids: object,
        *,
        produced_at: datetime,
    ) -> int:
        """Project one persisted recommendation run's track items (P15-S3-S2).

        The runtime inverse of the migration-0019 backfill B: each appearance of a canonical
        Catalog track as a persisted recommendation item bumps ``recommendation_count`` and
        stamps ``first_recommended_at`` / ``last_recommended_at`` from the run's
        ``produced_at``. Ordering follows B's own comparator -- true aware instant first, then
        verbatim lexical text as the same-instant tie-break -- so the incremental writer and a
        rebuild from ``recommendation_runs`` never disagree, including across mixed UTC
        offsets.

        Only existing rows are updated: a recommended track without a row (a library track,
        or an identity never catalog-bound) is skipped, never created -- the table stays
        catalog rows only, exactly like B's row-scoped UPDATE. One ``BEGIN IMMEDIATE``
        transaction covers the whole batch (per-run atomic); an exception rolls back and
        propagates (loud, never swallowed). Returns the number of tracks projected.
        """
        if not isinstance(track_ids, (list, tuple)):
            raise CatalogTrackStateError("track_ids must be a list or tuple")
        occurrences: dict[str, int] = {}
        for entry in track_ids:
            canonical_id = _require_canonical_id(entry)
            occurrences[canonical_id] = occurrences.get(canonical_id, 0) + 1
        if not occurrences:
            return 0
        if not isinstance(produced_at, datetime) or produced_at.tzinfo is None:
            raise CatalogTrackStateError("produced_at must be an aware datetime")
        stamp = produced_at.isoformat()
        projected = 0
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            for canonical_id, count in occurrences.items():
                row = self._connection.execute(
                    """SELECT first_recommended_at, last_recommended_at
                    FROM catalog_track_state WHERE canonical_id=?""",
                    (canonical_id,),
                ).fetchone()
                if row is None:
                    # Library target or an identity that is not catalog-bound: no row,
                    # and none is created here (the backfill's row scope is preserved).
                    continue
                first_text, last_text = row[0], row[1]
                if first_text is None or _earlier_by_backfill_order(
                    produced_at, stamp, first_text
                ):
                    first_text = stamp
                if last_text is None or _later_by_backfill_order(
                    produced_at, stamp, last_text
                ):
                    last_text = stamp
                self._connection.execute(
                    """UPDATE catalog_track_state SET
                        first_recommended_at=?, last_recommended_at=?,
                        recommendation_count=recommendation_count+?, updated_at=?
                    WHERE canonical_id=?""",
                    (first_text, last_text, count, stamp, canonical_id),
                )
                projected += 1
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")
        return projected

    def get_state(self, canonical_id: str) -> CatalogTrackState | None:
        """Read one track's long-term memory row, or ``None`` when no row exists."""
        canonical_id = _require_canonical_id(canonical_id)
        row = self._connection.execute(
            """SELECT canonical_id, source_system, first_discovered_at, last_discovered_at,
                      discovery_count, discovery_terms_json, first_recommended_at,
                      last_recommended_at, recommendation_count, updated_at
            FROM catalog_track_state WHERE canonical_id=?""",
            (canonical_id,),
        ).fetchone()
        if row is None:
            return None
        return _decode_state_row(row)

    def load_states(
        self, canonical_ids: object
    ) -> dict[str, CatalogTrackState]:
        """Read many tracks' rows in one query (P15-S3-S3A supply-facts read).

        Returns ``{canonical_id: state}`` for every existing row among the requested ids;
        unknown ids are dropped -- absence is "no memory", never "never happened" -- and
        duplicates are deduped. Read-only bulk counterpart of ``get_state`` that avoids one
        query per track when an execution needs the memory for a whole pool.
        """
        if not isinstance(canonical_ids, (list, tuple)):
            raise CatalogTrackStateError("canonical_ids must be a list or tuple")
        ids: list[str] = []
        seen: set[str] = set()
        for entry in canonical_ids:
            canonical_id = _require_canonical_id(entry)
            if canonical_id not in seen:
                seen.add(canonical_id)
                ids.append(canonical_id)
        if not ids:
            return {}
        states: dict[str, CatalogTrackState] = {}
        # Chunked to stay far below the SQLite bind-variable limit for full-pool reads.
        for start in range(0, len(ids), _LOAD_STATES_CHUNK):
            chunk = ids[start : start + _LOAD_STATES_CHUNK]
            placeholders = ", ".join("?" for _ in chunk)
            rows = self._connection.execute(
                f"""SELECT canonical_id, source_system, first_discovered_at,
                           last_discovered_at, discovery_count, discovery_terms_json,
                           first_recommended_at, last_recommended_at,
                           recommendation_count, updated_at
                FROM catalog_track_state WHERE canonical_id IN ({placeholders})""",
                tuple(chunk),
            ).fetchall()
            for row in rows:
                states[row[0]] = _decode_state_row(row)
        return states

    def find_states_by_term(self, term: object) -> tuple[CatalogTrackState, ...]:
        """Return rows whose bounded discovery-term memory contains ``term``.

        ``term`` is normalized with the same rules the discovery writer applies (strip +
        casefold + length bound); blank input yields the empty result. Rows without term
        memory (honestly backfilled rows) never match. Order is deterministic: most
        recently discovered first (NULL ``last_discovered_at`` -- no recorded discovery
        events -- sinks last), ``canonical_id`` descending as the tie-break.

        Read-only memory view only: this answers "which remembered tracks did this term
        yield", never a verdict on the live catalog's freshness -- catalog search reuse and
        the Known-vs-Fresh policy are P15-S3-S3 territory, not implemented here.
        """
        normalized = normalize_discovery_term(term)
        if normalized is None:
            return ()
        rows = self._connection.execute(
            """SELECT canonical_id, source_system, first_discovered_at, last_discovered_at,
                      discovery_count, discovery_terms_json, first_recommended_at,
                      last_recommended_at, recommendation_count, updated_at
            FROM catalog_track_state
            WHERE EXISTS (
                SELECT 1 FROM json_each(catalog_track_state.discovery_terms_json) term_row
                WHERE term_row.key = ?
            )
            ORDER BY last_discovered_at DESC, canonical_id DESC""",
            (normalized,),
        ).fetchall()
        return tuple(_decode_state_row(row) for row in rows)

    def count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM catalog_track_state").fetchone()
        return int(row[0])


def _require_canonical_id(canonical_id: object) -> str:
    if not isinstance(canonical_id, str):
        raise CatalogTrackStateError("canonical_id must be a non-empty string")
    try:
        validate_canonical_id(EntityType.TRACK, canonical_id)
    except ValueError as error:
        raise CatalogTrackStateError(str(error)) from error
    return canonical_id


def _require_source_system(source_system: object) -> str:
    if not isinstance(source_system, str) or source_system not in _CATALOG_SOURCE_SYSTEMS:
        raise CatalogTrackStateError(
            "source_system must be itunes_store or apple_music_catalog"
        )
    return source_system