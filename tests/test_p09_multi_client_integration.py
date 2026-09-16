"""P09.7: multi-client shared-state proof -- the Phase success criterion, scenario by scenario.

Two independent model/client identities (a Codex client and a DeepSeek client) interact through
the same :class:`SharedAgentService` over one SQLite store and must observe the same durable
state. The scenarios below map 1:1 to the Phase success criterion:

* A: both clients read the same canonical music facts, Preference state, Recommendation state,
  and Feedback / Learning state;
* B: an allowed mutation through client A is durably visible to client B after a full
  close/reopen through a fresh service instance (no direct database shortcut anywhere);
* C: no per-model state fork exists -- client/model identity is provenance in the request
  journal only, and the schema carries no per-model user-state tables or columns;
* D: permissions are enforced at the shared service boundary for every client, and a provider
  identity never elevates permissions;
* E: replay and idempotency contracts hold (journaled replay, fail-closed replay conflict,
  domain-level duplicate refusal, no double application);
* F: restart/reopen preserves shared state and request provenance;
* G: malformed / unsupported / unauthorized requests return stable fail-closed errors with no
  partial silent execution.

All state in these scenarios is created through the shared agent layer itself (record feedback,
apply learning, generate recommendation); only the initial canonical fixture is seeded, because
canonical ingestion is P02 territory, not an agent-tool operation.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from music_agent.agent_client import AgentClient
from music_agent.agent_contract import (
    AGENT_CONTRACT_VERSION,
    AgentClientIdentity,
    AgentToolOutcome,
    generate_request_id,
)
from music_agent.agent_permission import AgentClientPolicy, AgentClientRegistry
from music_agent.agent_service import SharedAgentService
from music_agent.identity import EntityType, ExternalIdentityKey
from music_agent.intent_repository import PendingIntentRepository
from music_agent.repository import CanonicalRepository
from music_agent.source_observation import ObservedValue
from music_agent.validation import validate_fixture
from music_agent.write_intent import (
    PendingIntent,
    RequirementRole,
    WriteOperation,
    WriteRequirement,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"

CODEX_CLIENT = "agt_11111111-1111-4111-8111-111111111111"
DEEPSEEK_CLIENT = "agt_22222222-2222-4222-8222-222222222222"
READ_ONLY_CLIENT = "agt_33333333-3333-4333-8333-333333333333"
UNREGISTERED_CLIENT = "agt_99999999-9999-4999-8999-999999999999"
TRACK_A = "trk_11111111-1111-4111-8111-111111111111"
TRACK_B = "trk_22222222-2222-4222-8222-222222222222"
INTENT_ID = "int_55555555-5555-4555-8555-555555555555"
FEEDBACK_A = "fbk_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
FEEDBACK_B = "fbk_bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
NOW = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)
ISO = "2026-08-16T00:00:00+00:00"


def registry() -> AgentClientRegistry:
    return AgentClientRegistry(
        {
            CODEX_CLIENT: AgentClientPolicy.FULL,
            DEEPSEEK_CLIENT: AgentClientPolicy.FULL,
            READ_ONLY_CLIENT: AgentClientPolicy.READ_ONLY,
        }
    )


def feedback_payload(track_id: str, feedback_id: str) -> dict:
    # P16-S1: observed_at is service-authoritative (no model payload key).
    return {
        "kind": "liked",
        "source_system": "recommendation_ui",
        "source_path": "card_actions",
        "target_id": track_id,
        "feedback_id": feedback_id,
    }


class MultiClientIntegrationTest(unittest.TestCase):
    """A + B + C + D + E + F + G -- one shared layer, two model clients, one durable store."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "shared.sqlite3"
        with open(FIXTURE_PATH, encoding="utf-8") as fixture_file:
            fixture = json.load(fixture_file)
        validate_fixture(fixture)
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
        with PendingIntentRepository(self.database_path) as repository:
            repository.save_intent(
                PendingIntent(
                    intent_id=INTENT_ID,
                    operation=WriteOperation.SET_FAVORITED,
                    requirements=(
                        WriteRequirement(
                            RequirementRole.TARGET,
                            TRACK_A,
                            ExternalIdentityKey("apple_music", EntityType.TRACK, "persist-001"),
                        ),
                    ),
                    requested_value=ObservedValue.value(True),
                )
            )
        self.service = SharedAgentService(self.database_path, clients=registry())
        self.codex = AgentClient(AgentClientIdentity(CODEX_CLIENT, "codex"), self.service)
        self.deepseek = AgentClient(
            AgentClientIdentity(DEEPSEEK_CLIENT, "deepseek"), self.service
        )
        self.read_only = AgentClient(
            AgentClientIdentity(READ_ONLY_CLIENT, "deepseek"), self.service
        )

    def tearDown(self) -> None:
        self.service.close()
        self.temporary_directory.cleanup()

    def ok(self, client: AgentClient, tool: str, payload: dict) -> dict:
        result = client.call(tool, payload, issued_at=NOW, completed_at=ISO)
        self.assertEqual(result.outcome, AgentToolOutcome.OK, result.error_message)
        return result.payload

    def refused(self, client: AgentClient, tool: str, payload: dict) -> object:
        result = client.call(tool, payload, issued_at=NOW, completed_at=ISO)
        self.assertNotEqual(result.outcome, AgentToolOutcome.OK)
        return result

    # --- A: shared read state ---------------------------------------------------

    def test_A_both_clients_read_the_same_shared_state(self) -> None:
        # State created entirely through the shared layer.
        self.ok(self.codex, "record_feedback", feedback_payload(TRACK_A, FEEDBACK_A))
        self.ok(self.codex, "apply_learning", {"feedback_id": FEEDBACK_A})
        recommendation = self.ok(
            self.codex,
            "generate_recommendation",
            {
                "target_ids": [TRACK_A, TRACK_B],
                "limit": 5,
                "source_system": "feedback_learning",
            },
        )
        self.assertEqual(recommendation["item_count"], 1)

        canonical_codex = self.ok(
            self.codex, "get_canonical_entity", {"canonical_id": TRACK_A}
        )
        canonical_deepseek = self.ok(
            self.deepseek, "get_canonical_entity", {"canonical_id": TRACK_A}
        )
        self.assertEqual(canonical_codex, canonical_deepseek)

        preference_codex = self.ok(
            self.codex,
            "query_track_preference",
            {"target_id": TRACK_A, "source_system": "feedback_learning"},
        )
        preference_deepseek = self.ok(
            self.deepseek,
            "query_track_preference",
            {"target_id": TRACK_A, "source_system": "feedback_learning"},
        )
        self.assertEqual(preference_codex, preference_deepseek)
        self.assertEqual(preference_deepseek["preference_state"], "positive")

        runs_codex = self.ok(self.codex, "list_recommendation_runs", {})
        runs_deepseek = self.ok(self.deepseek, "list_recommendation_runs", {})
        self.assertEqual(runs_codex, runs_deepseek)
        self.assertEqual(len(runs_codex["runs"]), 1)

        feedback_codex = self.ok(self.codex, "list_feedback_observations", {})
        feedback_deepseek = self.ok(self.deepseek, "list_feedback_observations", {})
        self.assertEqual(feedback_codex, feedback_deepseek)

        learning_codex = self.ok(self.codex, "list_learning_applications", {})
        learning_deepseek = self.ok(self.deepseek, "list_learning_applications", {})
        self.assertEqual(learning_codex, learning_deepseek)

    # --- B: cross-client visibility through a reopened service ----------------------

    def test_B_mutation_through_client_A_is_visible_to_client_B_after_reopen(self) -> None:
        self.ok(self.codex, "record_feedback", feedback_payload(TRACK_A, FEEDBACK_A))
        self.ok(self.codex, "apply_learning", {"feedback_id": FEEDBACK_A})
        self.ok(
            self.codex,
            "generate_recommendation",
            {
                "target_ids": [TRACK_A],
                "limit": 5,
                "source_system": "feedback_learning",
            },
        )
        self.service.close()

        reopened_service = SharedAgentService(self.database_path, clients=registry())
        try:
            deepseek_after_reopen = AgentClient(
                AgentClientIdentity(DEEPSEEK_CLIENT, "deepseek"), reopened_service
            )
            feedback = self.ok(deepseek_after_reopen, "list_feedback_observations", {})
            self.assertEqual(feedback["count"], 1)
            learning = self.ok(deepseek_after_reopen, "list_learning_applications", {})
            self.assertEqual(len(learning["applications"]), 1)
            preference = self.ok(
                deepseek_after_reopen,
                "query_track_preference",
                {"target_id": TRACK_A, "source_system": "feedback_learning"},
            )
            self.assertEqual(preference["preference_state"], "positive")
            runs = self.ok(deepseek_after_reopen, "list_recommendation_runs", {})
            self.assertEqual(len(runs["runs"]), 1)

            # And the reverse direction through the same reopened layer.
            self.ok(
                deepseek_after_reopen,
                "record_feedback",
                feedback_payload(TRACK_B, FEEDBACK_B),
            )
            codex_after_reopen = AgentClient(
                AgentClientIdentity(CODEX_CLIENT, "codex"), reopened_service
            )
            feedback = self.ok(codex_after_reopen, "list_feedback_observations", {})
            self.assertEqual(feedback["count"], 2)
        finally:
            reopened_service.close()

    # --- C: no per-model state fork ------------------------------------------------

    def test_C_no_per_model_state_fork_exists(self) -> None:
        self.ok(self.codex, "record_feedback", feedback_payload(TRACK_A, FEEDBACK_A))
        self.service.close()
        connection = sqlite3.connect(self.database_path)
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        finally:
            connection.close()
        for table in tables:
            self.assertNotIn("codex", table)
            self.assertNotIn("deepseek", table)
            self.assertNotIn("model", table)
        self.assertIn("agent_requests", tables)
        self.assertIn("feedback_observations", tables)

    # --- D: permission isolation ------------------------------------------------------

    def test_D_permissions_are_enforced_at_the_shared_boundary_for_every_client(self) -> None:
        # A read-only client is refused mutations whatever model it claims.
        denied = self.refused(
            self.read_only, "record_feedback", feedback_payload(TRACK_A, FEEDBACK_A)
        )
        self.assertEqual(denied.outcome, AgentToolOutcome.PERMISSION_DENIED)
        # An unregistered client id is unknown even when it claims a known model.
        unregistered = self.refused(
            AgentClient(
                AgentClientIdentity(UNREGISTERED_CLIENT, "codex"), self.service
            ),
            "get_agent_capabilities",
            {},
        )
        self.assertEqual(unregistered.outcome, AgentToolOutcome.UNKNOWN_CLIENT)
        # Live writes fail closed identically for both providers -- identity never elevates.
        for client in (self.codex, self.deepseek):
            refusal = self.refused(
                client, "execute_write_intent", {"intent_id": INTENT_ID}
            )
            self.assertEqual(refusal.outcome, AgentToolOutcome.NOT_EXECUTION_READY)
            self.assertEqual(refusal.error_code, "not_execution_ready")
        # The refused mutation left no trace: feedback history is still empty.
        listed = self.ok(self.deepseek, "list_feedback_observations", {})
        self.assertEqual(listed["count"], 0)

    # --- E: replay / idempotency ----------------------------------------------------------

    def test_E_replay_and_idempotency_contracts_hold(self) -> None:
        payload = feedback_payload(TRACK_A, FEEDBACK_A)
        request_id = generate_request_id()
        first = self.codex.call(
            "record_feedback", payload, request_id=request_id, issued_at=NOW, completed_at=ISO
        )
        self.assertEqual(first.outcome, AgentToolOutcome.OK)

        # Same request id + same payload: journaled replay, no second observation.
        replay = self.codex.call(
            "record_feedback", payload, request_id=request_id, issued_at=NOW, completed_at=ISO
        )
        self.assertTrue(replay.replayed)
        self.assertEqual(replay.payload, first.payload)
        self.assertEqual(
            self.ok(self.deepseek, "list_feedback_observations", {})["count"], 1
        )

        # Same request id + different payload: fail-closed replay conflict.
        conflict = self.codex.call(
            "record_feedback",
            {**payload, "kind": "disliked"},
            request_id=request_id,
            issued_at=NOW,
            completed_at=ISO,
        )
        self.assertEqual(conflict.outcome, AgentToolOutcome.REPLAY_CONFLICT)

        # Fresh request id + same semantic payload: domain dedup fails closed, no third copy.
        duplicate = self.codex.call(
            "record_feedback", payload, issued_at=NOW, completed_at=ISO
        )
        self.assertEqual(duplicate.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(duplicate.error_code, "duplicate_feedback_observation")
        self.assertEqual(
            self.ok(self.deepseek, "list_feedback_observations", {})["count"], 1
        )

        # Learning application replays without double-applying.
        apply_request_id = generate_request_id()
        applied = self.codex.call(
            "apply_learning",
            {"feedback_id": FEEDBACK_A},
            request_id=apply_request_id,
            issued_at=NOW,
            completed_at=ISO,
        )
        self.assertTrue(applied.payload["applied"])
        replayed = self.codex.call(
            "apply_learning",
            {"feedback_id": FEEDBACK_A},
            request_id=apply_request_id,
            issued_at=NOW,
            completed_at=ISO,
        )
        self.assertTrue(replayed.replayed)
        self.assertEqual(
            len(self.ok(self.deepseek, "list_learning_applications", {})["applications"]), 1
        )

    # --- F: restart durability ---------------------------------------------------------------

    def test_F_restart_preserves_shared_state_and_request_provenance(self) -> None:
        self.ok(self.codex, "record_feedback", feedback_payload(TRACK_A, FEEDBACK_A))
        self.ok(self.codex, "apply_learning", {"feedback_id": FEEDBACK_A})
        self.service.close()
        self.service = SharedAgentService(self.database_path, clients=registry())
        self.codex = AgentClient(AgentClientIdentity(CODEX_CLIENT, "codex"), self.service)
        self.deepseek = AgentClient(
            AgentClientIdentity(DEEPSEEK_CLIENT, "deepseek"), self.service
        )

        self.assertEqual(
            self.ok(self.deepseek, "list_feedback_observations", {})["count"], 1
        )
        preference = self.ok(
            self.deepseek,
            "query_track_preference",
            {"target_id": TRACK_A, "source_system": "feedback_learning"},
        )
        self.assertEqual(preference["preference_state"], "positive")
        from music_agent.agent_request_journal_repository import AgentRequestJournalRepository

        with AgentRequestJournalRepository(self.database_path) as journal:
            records = journal.list()
        self.assertGreaterEqual(len(records), 2)
        self.assertIn(CODEX_CLIENT, {record.request.client.client_id for record in records})
        self.assertIn("codex", {record.request.client.model_id for record in records})

    # --- G: error contract ----------------------------------------------------------------------

    def test_G_error_contract_is_stable_with_no_partial_execution(self) -> None:
        malformed = self.refused(
            self.codex, "query_track_preference", {"target_id": "rcm_1"}
        )
        self.assertEqual(malformed.outcome, AgentToolOutcome.INVALID_REQUEST)
        unknown_tool = self.refused(self.codex, "no_such_tool", {})
        self.assertEqual(unknown_tool.outcome, AgentToolOutcome.TOOL_NOT_SUPPORTED)
        self.assertEqual(unknown_tool.error_code, "tool_not_supported")
        unknown_client = self.refused(
            AgentClient(
                AgentClientIdentity(UNREGISTERED_CLIENT, "codex"), self.service
            ),
            "get_agent_capabilities",
            {},
        )
        self.assertEqual(unknown_client.outcome, AgentToolOutcome.UNKNOWN_CLIENT)
        missing_feedback = self.refused(
            self.codex, "interpret_feedback", {"feedback_id": FEEDBACK_A}
        )
        self.assertEqual(missing_feedback.outcome, AgentToolOutcome.EXECUTION_ERROR)
        self.assertEqual(missing_feedback.error_code, "feedback_not_found")
        from music_agent.agent_contract import AgentRequest

        future_version = AgentRequest(
            generate_request_id(),
            AgentClientIdentity(CODEX_CLIENT, "codex"),
            "get_agent_capabilities",
            {},
            NOW,
            contract_version=AGENT_CONTRACT_VERSION + 1,
        )
        unsupported = self.service.execute(future_version, completed_at=ISO)
        self.assertEqual(unsupported.outcome, AgentToolOutcome.INVALID_REQUEST)
        self.assertEqual(unsupported.error_code, "unsupported_contract_version")

        # None of the refusals left any state behind.
        self.assertEqual(
            self.ok(self.deepseek, "list_feedback_observations", {})["count"], 0
        )
        self.assertEqual(
            self.ok(self.deepseek, "list_recommendation_runs", {})["runs"], []
        )
        self.assertEqual(
            self.ok(self.deepseek, "list_learning_applications", {})["applications"], []
        )


if __name__ == "__main__":
    unittest.main()
