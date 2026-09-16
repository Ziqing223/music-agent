"""Durable SQLite persistence for recommendation run history (P07.5).

A ranked recommendation run (:class:`~music_agent.recommendation_contract.RecommendationResult`)
is persisted as one immutable, append-only row in ``recommendation_runs``. The authoritative
persisted unit is the canonical :func:`~music_agent.recommendation_contract.encode_recommendation_result`
JSON text carried in ``encoded_result``; ``run_id``, ``contract_version``, and ``produced_at`` are
mirrored as columns so runs can be enumerated cheaply and every row's mirror can be verified
against its payload. History order is newest-first by real insertion recency: ``created_at``
(UTC ``CURRENT_TIMESTAMP``, fixed-format TEXT) is the ordering key, so model-supplied
``produced_at`` values never influence ordering. History is isolated from every other table in
the store -- in particular from the P06 preference tables, which are read-only upstream state
that recommendation history never touches.

``save_result`` is the single write boundary: it validates the input is a
:class:`~music_agent.recommendation_contract.RecommendationResult`, encodes it with the canonical
interchange (never a hand-rolled JSON form), and inserts one row with mirrored columns taken
directly from the result fields. A duplicate ``run_id`` fails closed with
:class:`DuplicateRecommendationRunError`; there is no overwrite path.

``get_result`` and ``list_runs`` decode through the canonical
:func:`~music_agent.recommendation_contract.decode_recommendation_result` interchange and fail
closed with :class:`CorruptRecommendationHistoryError` on any row that does not decode or whose
mirrored columns disagree with its payload -- never a silent partial decode and never a silently
skipped run.

History is append-only by construction and by hard guarantee: the repository exposes no update or
delete method, and ``BEFORE UPDATE`` / ``BEFORE DELETE`` triggers on ``recommendation_runs``
(landed by the lead in migration 0012) make immutability a SQLite-enforced invariant.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from music_agent.recommendation_contract import (
    RecommendationContractValidationError,
    RecommendationResult,
    decode_recommendation_result,
    encode_recommendation_result,
)
from music_agent.repository import _open_store_connection


class RecommendationHistoryRepositoryError(ValueError):
    code = "recommendation_history_repository_error"


class DuplicateRecommendationRunError(RecommendationHistoryRepositoryError):
    """A run_id is already recorded; recommendation history has no overwrite path."""

    code = "duplicate_recommendation_run"


class CorruptRecommendationHistoryError(RecommendationHistoryRepositoryError):
    """A stored row does not decode to a RecommendationResult or its mirror disagrees."""

    code = "corrupt_recommendation_history"


class RecommendationHistoryRepository:
    """Persist immutable recommendation runs inside the shared SQLite store."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._connection = _open_store_connection(self.database_path)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> RecommendationHistoryRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()
        return int(row[0])

    def save_result(self, result: RecommendationResult) -> None:
        """Persist one recommendation run, failing closed on any duplicate run_id.

        ``result`` must be a :class:`RecommendationResult`; it is encoded with the canonical
        ``encode_recommendation_result`` interchange and stored with ``run_id`` /
        ``contract_version`` / ``produced_at`` mirrored from the result fields. Inside one
        ``BEGIN IMMEDIATE`` transaction the run_id is pre-checked and a duplicate fails closed
        with :class:`DuplicateRecommendationRunError` before anything is stored; the ``run_id``
        primary key remains the hard backstop, so an unexpected ``sqlite3.IntegrityError``
        propagates unchanged after the rollback.
        """
        result = _require_result(result)
        encoded = encode_recommendation_result(result)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self._connection.execute(
                "SELECT 1 FROM recommendation_runs WHERE run_id=?", (result.run_id,)
            ).fetchone()
            if existing is not None:
                raise DuplicateRecommendationRunError(
                    f"run {result.run_id!r} is already recorded"
                )
            self._connection.execute(
                """INSERT INTO recommendation_runs(
                    run_id, encoded_result, contract_version, produced_at
                ) VALUES (?, ?, ?, ?)""",
                (
                    result.run_id,
                    encoded,
                    result.contract_version,
                    result.produced_at.isoformat(),
                ),
            )
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def get_result(self, run_id: str) -> RecommendationResult | None:
        """Return the run recorded for ``run_id``, or ``None`` if no such run exists.

        A stored row that does not decode through ``decode_recommendation_result``, or whose
        mirrored columns disagree with its payload, fails closed with
        :class:`CorruptRecommendationHistoryError` -- never a silent partial decode.
        """
        run_id = _require_run_id(run_id)
        row = self._connection.execute(
            "SELECT * FROM recommendation_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        return None if row is None else _decode_row(row)

    def list_runs(self, *, limit: int | None = None) -> tuple[RecommendationResult, ...]:
        """Return stored runs in deterministic, newest-first insertion order.

        Runs are ordered by real insertion recency: the ``created_at`` column (UTC
        ``CURRENT_TIMESTAMP``, fixed ``YYYY-MM-DD HH:MM:SS`` format, second resolution) is the
        ordering key, so ordering never depends on model-supplied ``produced_at`` values. Equal
        ``created_at`` values break deterministically with ``run_id`` descending. The first tuple
        element is the most recently created run -- the deterministic resolution target for
        natural-language references such as "刚才推荐的".

        ``limit`` bounds the result to the most recent ``limit`` runs (the window is applied
        after ordering, so the newest runs always win); ``None`` (the default) returns every run
        exactly as before, so existing callers are unaffected. A corrupted row within the
        returned window fails closed with :class:`CorruptRecommendationHistoryError` rather than
        being silently skipped; rows beyond the window are never decoded.
        """
        sql = (
            """SELECT * FROM recommendation_runs
            ORDER BY created_at DESC, run_id DESC"""
        )
        if limit is None:
            rows = self._connection.execute(sql)
        else:
            rows = self._connection.execute(sql + " LIMIT ?", (limit,))
        return tuple(_decode_row(row) for row in rows)

    def newest_run_id(self) -> str | None:
        """The chronologically latest run id by REAL row insertion order.

        P19-T14-B (the web shell's reply door): :meth:`list_runs` order stays
        the frozen display contract, but its ``created_at`` key is second-
        resolution and same-second insertions break arbitrarily on the random
        ``run_id`` -- which is exactly wrong for the door's before/after
        identity compare inside one API call. The implicit SQLite rowid is
        pure insertion chronology (never model-influenced, unaffected by
        clock granularity, never reused -- the table is append-only by
        trigger), so this read identifies the run persisted most recently.
        Returns ``None`` when no run was ever recorded.
        """
        row = self._connection.execute(
            "SELECT run_id FROM recommendation_runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        return None if row is None else row["run_id"]


def _require_result(result: object) -> RecommendationResult:
    if not isinstance(result, RecommendationResult):
        raise RecommendationHistoryRepositoryError("result must be a RecommendationResult")
    return result


def _require_run_id(run_id: object) -> str:
    if not isinstance(run_id, str) or run_id == "":
        raise RecommendationHistoryRepositoryError("run_id must be a non-empty string")
    return run_id


def _decode_row(row: sqlite3.Row) -> RecommendationResult:
    """Decode one stored row through the canonical interchange, verifying its mirrored columns."""
    try:
        result = decode_recommendation_result(row["encoded_result"])
    except RecommendationContractValidationError as error:
        raise CorruptRecommendationHistoryError(
            f"corrupt recommendation run row {row['run_id']!r}: {error}"
        ) from error
    if (
        result.run_id != row["run_id"]
        or result.contract_version != row["contract_version"]
        or result.produced_at.isoformat() != row["produced_at"]
    ):
        raise CorruptRecommendationHistoryError(
            f"corrupt recommendation run row {row['run_id']!r}: "
            "mirrored columns disagree with the encoded result"
        )
    return result
