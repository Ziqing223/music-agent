"""Write pipeline: source observations into durable preference evidence (P06 integration).

This module is the *write* half of the P06 end-to-end integration. It maps an already-resolved
``SourceObservation`` for a canonical Track onto the durable preference source-of-truth owned by
:mod:`~music_agent.preference_persistence_repository`, and nothing else. It is a thin adapter: it
decides *which* source fields are preference signals, builds a deterministic
:class:`~music_agent.preference_persistence.SignalIdentity` for each, and delegates every write to
``PreferencePersistenceRepository.record_observation``. It never re-interprets a value into a
preference direction (S1 / S3 own inference), never computes familiarity or confidence, and never
touches the canonical entity tables.

Signal scope
------------

The current P06 core persists exactly four Track-level signals:

``favorited`` / ``disliked`` / ``rating``
    Directional preference evidence consumed by S1 + S3 at query time.

``play_count``
    Historical exposure consumed by S4 (familiarity) at query time.

``skip_count`` is deliberately *not* persisted, and ``last_played_at`` / ``added_to_library_at``
remain structurally possible (the persistence schema's ``signal_path`` is a free-form string) but
are not written by this pipeline, so they stay absent from the durable preference store.

Three-state preservation
------------------------

``MISSING`` / ``NULL`` / ``VALUE`` are preserved exactly as the source reported them. A field that
is present in the observation (even as ``MISSING``) is recorded; a field that is entirely absent
from the observation's ``fields`` mapping is skipped and remains absent. The repository's frozen
semantics then apply unchanged: only a ``VALUE`` produces or changes a semantic value, and
``MISSING`` / ``NULL`` update head metadata only, never creating an evidence revision.

Transaction / error boundary
----------------------------

Each signal is recorded in its own ``BEGIN IMMEDIATE`` transaction inside ``record_observation``.
There is no cross-signal transaction: a failure in one signal's write leaves the already-committed
signals in place and does not roll them back. This is safe because the pipeline is idempotent --
re-running it reproduces confirmations (``revision is None``) rather than duplicate evidence -- and
because the preference tables are isolated from ``canonical_entities``, so no preference write
failure can corrupt canonical library state.
"""

from __future__ import annotations

from datetime import datetime, timezone

from music_agent.identity import EntityType
from music_agent.preference_attribution import (
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_persistence import (
    DIRECT_OBSERVATION_PROVENANCE,
    RecordObservationOutcome,
    SignalIdentity,
)
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.source_observation import SourceObservation


class PreferenceIngestionError(ValueError):
    code = "preference_ingestion_error"


# Ordered source field path -> signal path mapping. Order is deterministic and defines the order of
# the returned outcomes: favorited, disliked, rating, play_count.
SUPPORTED_PREFERENCE_SIGNALS: tuple[tuple[str, str], ...] = (
    ("library_state.favorited", "favorited"),
    ("library_state.disliked", "disliked"),
    ("library_state.rating", "rating"),
    ("library_state.play_count", "play_count"),
)


def ingest_track_observation(
    repository: PreferencePersistenceRepository,
    observation: SourceObservation,
    *,
    observed_at: str | None = None,
    event_at: str | None = None,
    provenance: str = DIRECT_OBSERVATION_PROVENANCE,
) -> tuple[RecordObservationOutcome, ...]:
    """Persist one Track ``SourceObservation`` into the preference source-of-truth.

    ``repository`` is the durable preference store. ``observation`` must be a ``TRACK``
    ``SourceObservation``; any other entity type fails closed because the supported preference
    signals are Track-level. ``observed_at`` defaults to the current instant (resolved once for the
    whole observation so all signals share one observation instant); ``event_at`` is the optional
    underlying event time passed through unchanged; ``provenance`` labels the observation.

    Each supported signal that is present in ``observation.fields`` (in any of ``MISSING`` /
    ``NULL`` / ``VALUE``) is recorded via :meth:`PreferencePersistenceRepository.record_observation`.
    A field absent from ``observation.fields`` is skipped and remains absent. The returned outcomes
    are in the fixed ``SUPPORTED_PREFERENCE_SIGNALS`` order, each outcome's ``head.identity`` naming
    the signal it belongs to.
    """
    if not isinstance(repository, PreferencePersistenceRepository):
        raise PreferenceIngestionError("repository must be a PreferencePersistenceRepository")
    if not isinstance(observation, SourceObservation):
        raise PreferenceIngestionError("observation must be a SourceObservation")
    if observation.entity_type is not EntityType.TRACK:
        raise PreferenceIngestionError(
            f"preference ingestion supports only Track observations, not "
            f"{observation.entity_type.value}"
        )

    resolved_observed_at = (
        datetime.now(timezone.utc).isoformat() if observed_at is None else observed_at
    )
    target = PreferenceTargetReference(PreferenceTargetKind.TRACK, observation.canonical_id)

    outcomes: list[RecordObservationOutcome] = []
    for source_path, signal_path in SUPPORTED_PREFERENCE_SIGNALS:
        observed = observation.fields.get(source_path)
        if observed is None:
            continue
        identity = SignalIdentity(target, observation.source_system, signal_path)
        outcomes.append(
            repository.record_observation(
                identity,
                observed,
                observed_at=resolved_observed_at,
                event_at=event_at,
                provenance=provenance,
            )
        )
    return tuple(outcomes)
