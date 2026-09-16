"""P09.4: the durable agent request journal -- append-only replay safety.

These tests prove the replay-safety backbone of the shared agent layer: fresh stores reach
schema v15 with the ``agent_requests`` table, a recorded request round-trips at full fidelity
(client label included) with deterministic chronological ordering, duplicate request ids fail
closed before write, request/result identity mismatch fails closed, the immutability triggers
make out-of-band updates impossible, corrupted or mirror-disagreeing rows fail closed on read,
and a v14 store upgrades to v15 without changing any existing P06 preference, P07 recommendation,
or P08 feedback state.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from music_agent.agent_contract import (
    AgentClientIdentity,
    AgentContractValidationError,
    AgentRequest,
    AgentToolOutcome,
    AgentToolResult,
)
from music_agent.agent_request_journal_repository import (
    AgentRequestJournalRepository,
    AgentRequestJournalRepositoryError,
    CorruptAgentRequestJournalError,
    DuplicateAgentRequestError,
)
from music_agent.feedback_contract import (
    FeedbackKind,
    FeedbackSourceReference,
    assemble_feedback_observation,
)
from music_agent.feedback_history_repository import FeedbackHistoryRepository
from music_agent.preference_attribution import (
    DerivedPreference,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_persistence import SignalIdentity
from music_agent.preference_persistence_repository import PreferencePersistenceRepository
from music_agent.preference_strength import PreferenceState, PreferenceStrength
from music_agent.recommendation_contract import (
    Candidate,
    CandidateSourceReference,
    PreferenceInput,
    RecommendationContext,
    RecommendationItem,
    RecommendationRequest,
    RecommendedItemKind,
    ScoreBreakdown,
    ScoreComponent,
    assemble_recommendation_result,
)
from music_agent.recommendation_history_repository import RecommendationHistoryRepository
from music_agent.repository import CURRENT_SCHEMA_VERSION, MIGRATIONS
from music_agent.source_observation import ObservedValue

CLIENT_ID = "agt_11111111-1111-4111-8111-111111111111"
REQUEST_ID = "req_22222222-2222-4222-8222-222222222222"
REQUEST_ID_2 = "req_33333333-3333-4333-8333-333333333333"
REQUEST_ID_3 = "req_44444444-4444-4444-8444-444444444444"
TRACK_ID = "trk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
RUN_ID = "rcm_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
CANDIDATE_ID = "cnd_cccccccc-cccc-4ccc-8ccc-cccccccccccc"
FEEDBACK_ID = "fbk_dddddddd-dddd-4ddd-8ddd-dddddddddddd"
NOW = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)


def client(model_id: str = "codex", label: str | None = "codex-chat adapter") -> AgentClientIdentity:
    return AgentClientIdentity(CLIENT_ID, model_id, label)


def request(
    request_id: str = REQUEST_ID,
    tool: str = "get_canonical_entity",
    payload: dict | None = None,
    issued_at: datetime = NOW,
) -> AgentRequest:
    return AgentRequest(
        request_id,
        client(),
        tool,
        {} if payload is None else payload,
        issued_at,
    )


def ok_result(
    request_id: str = REQUEST_ID,
    tool: str = "get_canonical_entity",
    payload: dict | None = None,
    completed_at: datetime = NOW,
) -> AgentToolResult:
    return AgentToolResult(
        request_id, tool, AgentToolOutcome.OK, {} if payload is None else payload, None, None, completed_at
    )


def error_result(
    request_id: str = REQUEST_ID,
    tool: str = "get_canonical_entity",
    code: str = "permission_denied",
) -> AgentToolResult:
    return AgentToolResult(
        request_id, tool, AgentToolOutcome.PERMISSION_DENIED, None, code, code, NOW
    )


class AgentRequestJournalRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_fresh_database_reaches_current_version_with_journal_table(self) -> None:
        with AgentRequestJournalRepository(self.database_path) as repository:
            self.assertEqual(repository.schema_version, 19)
            self.assertEqual(CURRENT_SCHEMA_VERSION, 19)
            tables = {
                row[0]
                for row in repository._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        self.assertIn("agent_requests", tables)

    def test_record_get_round_trip_preserves_full_fidelity(self) -> None:
        envelope = request(payload={"canonical_id": TRACK_ID})
        result = ok_result(payload={"entity": {"id": TRACK_ID}})
        with AgentRequestJournalRepository(self.database_path) as repository:
            repository.record(envelope, result)
            record = repository.get(REQUEST_ID)
        self.assertIsNotNone(record)
        self.assertEqual(record.request, envelope)
        self.assertEqual(record.result, result)
        # The client label survives readback (stored in the full canonical request text).
        self.assertEqual(record.request.client.label, "codex-chat adapter")

    def test_missing_request_returns_none(self) -> None:
        with AgentRequestJournalRepository(self.database_path) as repository:
            self.assertIsNone(repository.get(REQUEST_ID))

    def test_get_requires_the_request_namespace(self) -> None:
        with AgentRequestJournalRepository(self.database_path) as repository:
            with self.assertRaises(AgentContractValidationError):
                repository.get("rcm_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")

    def test_list_orders_by_issued_at_with_request_id_tie_break(self) -> None:
        later = NOW + timedelta(hours=1)
        with AgentRequestJournalRepository(self.database_path) as repository:
            repository.record(
                request(REQUEST_ID_2, issued_at=later),
                ok_result(REQUEST_ID_2, completed_at=later),
            )
            repository.record(request(REQUEST_ID), ok_result())
            records = repository.list()
        self.assertEqual(
            [record.request.request_id for record in records], [REQUEST_ID, REQUEST_ID_2]
        )
        # Same instant: request_id ascending tie-break (REQUEST_ID < REQUEST_ID_3 at NOW),
        # with the later-issued row still last.
        with AgentRequestJournalRepository(self.database_path) as repository:
            repository.record(
                request(REQUEST_ID_3, issued_at=NOW), ok_result(REQUEST_ID_3)
            )
            records = repository.list()
        self.assertEqual(
            [record.request.request_id for record in records],
            [REQUEST_ID, REQUEST_ID_3, REQUEST_ID_2],
        )

    def test_duplicate_request_id_fails_closed(self) -> None:
        with AgentRequestJournalRepository(self.database_path) as repository:
            repository.record(request(), ok_result())
            with self.assertRaises(DuplicateAgentRequestError):
                repository.record(request(), ok_result())

    def test_refusal_outcomes_are_journaled_too(self) -> None:
        with AgentRequestJournalRepository(self.database_path) as repository:
            repository.record(request(), error_result())
            record = repository.get(REQUEST_ID)
        self.assertEqual(record.result.outcome, AgentToolOutcome.PERMISSION_DENIED)
        self.assertEqual(record.result.error_code, "permission_denied")
        self.assertIsNone(record.result.payload)

    def test_request_result_identity_mismatch_fails_closed(self) -> None:
        with AgentRequestJournalRepository(self.database_path) as repository:
            with self.assertRaises(AgentRequestJournalRepositoryError):
                repository.record(request(), ok_result(REQUEST_ID_2))
            with self.assertRaises(AgentRequestJournalRepositoryError):
                repository.record(
                    request(), ok_result(tool="query_track_preference")
                )
            with self.assertRaises(AgentRequestJournalRepositoryError):
                repository.record("not-a-request", ok_result())  # type: ignore[arg-type]
            with self.assertRaises(AgentRequestJournalRepositoryError):
                repository.record(request(), "not-a-result")  # type: ignore[arg-type]

    def test_immutability_triggers_exist(self) -> None:
        with AgentRequestJournalRepository(self.database_path) as repository:
            triggers = {
                row[0]
                for row in repository._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                )
            }
        self.assertIn("trg_agent_requests_immutable_update", triggers)
        self.assertIn("trg_agent_requests_immutable_delete", triggers)

    def test_corrupted_row_decoding_fails_closed(self) -> None:
        with AgentRequestJournalRepository(self.database_path) as repository:
            repository.record(request(), ok_result())
            # Out-of-band tampering: violate a mirror column.
            with self.assertRaises(sqlite3.IntegrityError):
                repository._connection.execute(
                    "UPDATE agent_requests SET client_id = 'agt_99999999-9999-4999-8999-999999999999' WHERE request_id = ?",
                    (REQUEST_ID,),
                )
        with AgentRequestJournalRepository(self.database_path) as repository:
            # The trigger refused the update; the row still reads back faithfully.
            record = repository.get(REQUEST_ID)
            self.assertEqual(record.request.client.client_id, CLIENT_ID)

    def test_mirror_disagreement_fails_closed_on_list(self) -> None:
        with AgentRequestJournalRepository(self.database_path) as repository:
            repository.record(request(), error_result())
            # Bypass immutability at the trigger level is impossible, so simulate a
            # disagreeing mirror by inserting a row whose outcome mirror lies.
            repository._connection.execute(
                """INSERT INTO agent_requests(
                    request_id, client_id, model_id, tool_name, contract_version,
                    request_text, payload_text, outcome, result_text, error_code,
                    error_message, issued_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    REQUEST_ID_2,
                    CLIENT_ID,
                    "codex",
                    "get_canonical_entity",
                    1,
                    _encode_request_like(REQUEST_ID_2),
                    "{}",
                    "ok",
                    _encode_error_result_like(REQUEST_ID_2),
                    "permission_denied",
                    "permission_denied",
                    NOW.isoformat(),
                    NOW.isoformat(),
                ),
            )
            with self.assertRaises(CorruptAgentRequestJournalError):
                repository.list()
            with self.assertRaises(CorruptAgentRequestJournalError):
                repository.get(REQUEST_ID_2)


def _encode_request_like(request_id: str) -> str:
    from music_agent.agent_contract import encode_agent_request

    return encode_agent_request(
        AgentRequest(
            request_id,
            AgentClientIdentity(CLIENT_ID, "codex", None),
            "get_canonical_entity",
            {},
            NOW,
        )
    )


def _encode_error_result_like(request_id: str) -> str:
    from music_agent.agent_contract import encode_agent_tool_result

    return encode_agent_tool_result(
        AgentToolResult(
            request_id,
            "get_canonical_entity",
            AgentToolOutcome.PERMISSION_DENIED,
            None,
            "permission_denied",
            "permission_denied",
            NOW,
        )
    )


class HistoricalUpgradeTest(unittest.TestCase):
    """A v14 store (the pre-P09 baseline) upgrades to v15 without changing prior state."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "canonical.sqlite3"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_v14_store_upgrades_to_current_without_changing_prior_state(self) -> None:
        identity = SignalIdentity(track_target(), "apple_music", "favorited")
        with patch("music_agent.repository.MIGRATIONS", MIGRATIONS[:14]):
            with PreferencePersistenceRepository(self.database_path) as repository:
                repository.record_observation(
                    identity, ObservedValue.value(True), observed_at=NOW.isoformat()
                )
                self.assertEqual(repository.schema_version, 14)
            with RecommendationHistoryRepository(self.database_path) as repository:
                repository.save_result(recommendation_result())
            with FeedbackHistoryRepository(self.database_path) as repository:
                repository.save_observation(feedback_observation())

        with AgentRequestJournalRepository(self.database_path) as journal:
            self.assertEqual(journal.schema_version, 19)
            self.assertEqual(CURRENT_SCHEMA_VERSION, 19)
            journal.record(request(), ok_result())
            self.assertEqual(journal.get(REQUEST_ID).request, request())

        # Prior P06 / P07 / P08 state survives the upgrade untouched.
        with PreferencePersistenceRepository(self.database_path) as repository:
            head = repository.get_head(identity)
            self.assertEqual(head.current_revision_sequence, 1)
            self.assertIs(head.current_semantic_value, True)
        with RecommendationHistoryRepository(self.database_path) as repository:
            self.assertEqual(
                repository.get_result(RUN_ID), recommendation_result()
            )
        with FeedbackHistoryRepository(self.database_path) as repository:
            self.assertEqual(
                repository.get_observation(FEEDBACK_ID), feedback_observation()
            )


def track_target() -> PreferenceTargetReference:
    return PreferenceTargetReference(PreferenceTargetKind.TRACK, TRACK_ID)


def recommendation_result() -> object:
    strength = PreferenceStrength(PreferenceState.POSITIVE, 0.9)
    context = RecommendationContext(
        NOW, (PreferenceInput.from_direct(DerivedPreference(track_target(), strength)),)
    )
    recommendation_request = RecommendationRequest(context, RecommendedItemKind.TRACK, 5)
    candidate = Candidate(
        CANDIDATE_ID,
        track_target(),
        CandidateSourceReference("candidate_gen", "preference_match"),
    )
    item = RecommendationItem(
        candidate, ScoreBreakdown(0.9, (ScoreComponent("preference_match", 0.9),))
    )
    return assemble_recommendation_result(
        recommendation_request, (item,), run_id=RUN_ID, produced_at=NOW
    )


def feedback_observation() -> object:
    return assemble_feedback_observation(
        feedback_id=FEEDBACK_ID,
        kind=FeedbackKind.LIKED,
        source=FeedbackSourceReference("recommendation_ui", "card_actions"),
        observed_at=NOW,
        target=track_target(),
    )


if __name__ == "__main__":
    unittest.main()
