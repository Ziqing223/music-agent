"""Durable SQLite persistence for feedback observation history (P08.2).

A feedback observation (:class:`~music_agent.feedback_contract.FeedbackObservation`) is persisted
as one immutable, append-only row in ``feedback_observations``. The authoritative persisted unit
is the canonical :func:`~music_agent.feedback_contract.encode_feedback_observation` JSON text
carried in ``encoded_observation``; ``feedback_id``, ``kind``, ``contract_version``,
``observed_at``, and ``duplicate_key`` are mirrored as columns so observations can be enumerated,
deduplicated, and ordered without decoding, and every row's mirror can be verified against its
payload. History order is chronological, computed from the decoded timezone-aware ``observed_at``
values: TEXT lexicographic order cannot express cross-offset chronology. History is isolated from
every other table in the store -- in particular from the P06 preference tables and the P07
recommendation history, which are read-only upstream state that feedback history never touches.

``save_observation`` is the single write boundary: it validates the input is a
:class:`~music_agent.feedback_contract.FeedbackObservation`, encodes it with the canonical
interchange (never a hand-rolled JSON form), computes the duplicate key with
:func:`encode_duplicate_key`, and inserts one row with mirrored columns taken directly from the
observation. Duplicates fail closed with :class:`DuplicateFeedbackObservationError` before
anything is stored: a reused ``feedback_id`` is rejected, and a second observation whose duplicate
key equals an already-recorded row's is rejected as the same observed event even when it carries a
different ``feedback_id``. The ``feedback_id`` primary key and the UNIQUE ``duplicate_key`` index
(landed by migration 0013) remain the hard backstops, so an unexpected ``sqlite3.IntegrityError``
propagates unchanged after the rollback. No confirmation-counting policy is applied here: what a
learning slice *does* with a rejected duplicate (treat it as confirmation, drop it) is deferred by
the P08.1 contract and this repository only fails closed.

``get_observation`` and ``list_observations`` decode through the canonical
:func:`~music_agent.feedback_contract.decode_feedback_observation` interchange and fail closed
with :class:`CorruptFeedbackHistoryError` on any row that does not decode or whose mirrored
columns disagree with its payload -- never a silent partial decode and never a silently skipped
observation.

History is append-only by construction and by hard guarantee: the repository exposes no update or
delete method, and ``BEFORE UPDATE`` / ``BEFORE DELETE`` triggers on ``feedback_observations``
(landed by migration 0013) make immutability a SQLite-enforced invariant.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from music_agent.feedback_contract import (
    FeedbackContractValidationError,
    FeedbackObservation,
    decode_feedback_observation,
    encode_feedback_observation,
    feedback_duplicate_key,
)
from music_agent.preference_attribution import PreferenceTargetReference
from music_agent.repository import _open_store_connection


class FeedbackHistoryRepositoryError(ValueError):
    code = "feedback_history_repository_error"


class DuplicateFeedbackObservationError(FeedbackHistoryRepositoryError):
    """The observation, or the event its duplicate key identifies, is already recorded."""

    code = "duplicate_feedback_observation"


class CorruptFeedbackHistoryError(FeedbackHistoryRepositoryError):
    """A stored row does not decode to a FeedbackObservation or its mirror disagrees."""

    code = "corrupt_feedback_history"


def encode_duplicate_key(observation: FeedbackObservation) -> str:
    """Return the canonical TEXT form of an observation's :func:`feedback_duplicate_key`.

    The text is a deterministic JSON array of the key's normalized elements. Datetimes are
    canonicalized to UTC before encoding, so two timezone-aware datetimes that are equal as
    instants -- and therefore equal in the contract's duplicate key -- always encode to the same
    text. This is the value stored in the ``duplicate_key`` column whose UNIQUE index makes
    "one stored row per observed event" a hard SQLite guarantee.
    """
    if not isinstance(observation, FeedbackObservation):
        raise FeedbackHistoryRepositoryError(
            "observation must be a FeedbackObservation"
        )
    key = feedback_duplicate_key(observation)
    return json.dumps(
        _key_to_json(key), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def _key_to_json(value: object) -> object:
    if isinstance(value, str):
        return value
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, PreferenceTargetReference):
        return {"kind": value.kind.value, "target_id": value.target_id}
    if hasattr(value, "__dataclass_fields__"):
        return {
            field: _key_to_json(getattr(value, field))
            for field in sorted(value.__dataclass_fields__)
        }
    if isinstance(value, tuple):
        return [_key_to_json(item) for item in value]
    return value


class FeedbackHistoryRepository:
    """Persist immutable feedback observations inside the shared SQLite store."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._connection = _open_store_connection(self.database_path)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> FeedbackHistoryRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).fetchone()
        return int(row[0])

    def save_observation(self, observation: FeedbackObservation) -> None:
        """Persist one feedback observation, failing closed on any duplicate.

        ``observation`` must be a :class:`FeedbackObservation`; it is encoded with the canonical
        ``encode_feedback_observation`` interchange and stored with ``feedback_id`` / ``kind`` /
        ``contract_version`` / ``observed_at`` / ``duplicate_key`` mirrored from the observation.
        Inside one ``BEGIN IMMEDIATE`` transaction a reused ``feedback_id`` and an already-recorded
        duplicate key are each pre-checked and fail closed with
        :class:`DuplicateFeedbackObservationError` before anything is stored; the ``feedback_id``
        primary key and the UNIQUE ``duplicate_key`` index remain the hard backstops, so an
        unexpected ``sqlite3.IntegrityError`` propagates unchanged after the rollback.
        """
        observation = _require_observation(observation)
        encoded = encode_feedback_observation(observation)
        duplicate_key = encode_duplicate_key(observation)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            existing_id = self._connection.execute(
                "SELECT 1 FROM feedback_observations WHERE feedback_id=?",
                (observation.feedback_id,),
            ).fetchone()
            if existing_id is not None:
                raise DuplicateFeedbackObservationError(
                    f"feedback {observation.feedback_id!r} is already recorded"
                )
            existing_event = self._connection.execute(
                "SELECT feedback_id FROM feedback_observations WHERE duplicate_key=?",
                (duplicate_key,),
            ).fetchone()
            if existing_event is not None:
                raise DuplicateFeedbackObservationError(
                    f"the event described by observation {observation.feedback_id!r} is already "
                    f"recorded under feedback {existing_event['feedback_id']!r}"
                )
            self._connection.execute(
                """INSERT INTO feedback_observations(
                    feedback_id, encoded_observation, kind, contract_version,
                    observed_at, duplicate_key
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    observation.feedback_id,
                    encoded,
                    observation.kind.value,
                    observation.contract_version,
                    observation.observed_at.isoformat(),
                    duplicate_key,
                ),
            )
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def get_observation(self, feedback_id: str) -> FeedbackObservation | None:
        """Return the observation recorded for ``feedback_id``, or ``None`` if absent.

        A stored row that does not decode through ``decode_feedback_observation``, or whose
        mirrored columns disagree with its payload, fails closed with
        :class:`CorruptFeedbackHistoryError` -- never a silent partial decode.
        """
        feedback_id = _require_feedback_id(feedback_id)
        row = self._connection.execute(
            "SELECT * FROM feedback_observations WHERE feedback_id=?", (feedback_id,)
        ).fetchone()
        return None if row is None else _decode_row(row)

    def list_observations(self) -> tuple[FeedbackObservation, ...]:
        """Return every stored observation in deterministic, chronological history order.

        Observations are ordered by ``observed_at`` ascending, compared as timezone-aware
        datetimes (so observations recorded under different UTC offsets are ordered
        chronologically, never by lexicographic TEXT comparison of the stored timestamps), with
        ``feedback_id`` ascending as the documented tie-break for equal ``observed_at`` values.
        The SQL fetch keeps a deterministic row order; the returned tuple is sorted
        chronologically in Python. A corrupted row anywhere in the history fails closed with
        :class:`CorruptFeedbackHistoryError` rather than being silently skipped.
        """
        rows = self._connection.execute(
            """SELECT * FROM feedback_observations
            ORDER BY observed_at ASC, feedback_id ASC"""
        )
        observations = tuple(_decode_row(row) for row in rows)
        return tuple(
            sorted(observations, key=lambda obs: (obs.observed_at, obs.feedback_id))
        )


def _require_observation(observation: object) -> FeedbackObservation:
    if not isinstance(observation, FeedbackObservation):
        raise FeedbackHistoryRepositoryError(
            "observation must be a FeedbackObservation"
        )
    return observation


def _require_feedback_id(feedback_id: object) -> str:
    if not isinstance(feedback_id, str) or feedback_id == "":
        raise FeedbackHistoryRepositoryError(
            "feedback_id must be a non-empty string"
        )
    return feedback_id


def _decode_row(row: sqlite3.Row) -> FeedbackObservation:
    """Decode one stored row through the canonical interchange, verifying its mirrored columns."""
    try:
        observation = decode_feedback_observation(row["encoded_observation"])
    except FeedbackContractValidationError as error:
        raise CorruptFeedbackHistoryError(
            f"corrupt feedback observation row {row['feedback_id']!r}: {error}"
        ) from error
    if (
        observation.feedback_id != row["feedback_id"]
        or observation.kind.value != row["kind"]
        or observation.contract_version != row["contract_version"]
        or observation.observed_at.isoformat() != row["observed_at"]
        or encode_duplicate_key(observation) != row["duplicate_key"]
    ):
        raise CorruptFeedbackHistoryError(
            f"corrupt feedback observation row {row['feedback_id']!r}: "
            "mirrored columns disagree with the encoded observation"
        )
    return observation
