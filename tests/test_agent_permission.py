"""P09.3: the agent permission decision point -- policy above raw capability.

These tests prove the model-independent permission boundary: policy binds to the registered
``agt_`` client id and never to model metadata (an unregistered id stays unknown no matter what
model it claims; a registered id keeps its policy no matter what model it claims), client
policies gate read/mutate/live-write classes before any execution, live writes delegate to the
sealed ``WRITE_CAPABILITY_MATRIX`` + ``is_execution_ready`` predicate (so no write is
execution-ready today and the decision can never bypass it), and the capability summary projects
the gate read-only for the agent surface.
"""

from __future__ import annotations

import unittest

from music_agent.agent_contract import AgentToolOutcome
from music_agent.agent_permission import (
    AgentClientPolicy,
    AgentClientRegistry,
    AgentPermissionValidationError,
    PermissionDenial,
    decide_agent_permission,
    write_capability_summary,
)
from music_agent.agent_tools import AgentToolPermissionClass
from music_agent.write_intent import (
    WRITE_CAPABILITY_MATRIX,
    DomainPermission,
    WriteCapability,
    WriteOperation,
)

CLIENT_FULL = "agt_11111111-1111-4111-8111-111111111111"
CLIENT_READ_ONLY = "agt_22222222-2222-4222-8222-222222222222"
CLIENT_NONE = "agt_33333333-3333-4333-8333-333333333333"
TRACK = WriteOperation.SET_FAVORITED


class AgentClientRegistryTest(unittest.TestCase):
    def test_registry_resolves_policy_by_client_id(self) -> None:
        registry = AgentClientRegistry({CLIENT_FULL: AgentClientPolicy.FULL})
        self.assertIs(registry.policy_for(CLIENT_FULL), AgentClientPolicy.FULL)
        self.assertIsNone(registry.policy_for("agt_99999999-9999-4999-8999-999999999999"))

    def test_registry_fails_closed_on_invalid_registrations(self) -> None:
        with self.assertRaises(AgentPermissionValidationError):
            AgentClientRegistry({"trk_11111111-1111-4111-8111-111111111111": AgentClientPolicy.FULL})
        with self.assertRaises(AgentPermissionValidationError):
            AgentClientRegistry({CLIENT_FULL: "full"})  # type: ignore[dict-item]

    def test_policy_lookup_never_considers_model_metadata(self) -> None:
        # The registry is keyed by client id only; there is no model dimension to consult.
        registry = AgentClientRegistry({CLIENT_FULL: AgentClientPolicy.READ_ONLY})
        for model_id in ("codex", "deepseek", "claude"):
            self.assertIs(registry.policy_for(CLIENT_FULL), AgentClientPolicy.READ_ONLY)


class DecideAgentPermissionTest(unittest.TestCase):
    def test_full_client_is_permitted_for_reads_and_mutations(self) -> None:
        self.assertIsNone(
            decide_agent_permission(
                AgentToolPermissionClass.READ, AgentClientPolicy.FULL
            )
        )
        self.assertIsNone(
            decide_agent_permission(
                AgentToolPermissionClass.MUTATE, AgentClientPolicy.FULL
            )
        )

    def test_read_only_client_is_denied_for_mutations_and_live_writes(self) -> None:
        for permission_class in (
            AgentToolPermissionClass.MUTATE,
            AgentToolPermissionClass.LIVE_WRITE,
        ):
            denial = decide_agent_permission(permission_class, AgentClientPolicy.READ_ONLY)
            self.assertEqual(denial.outcome, AgentToolOutcome.PERMISSION_DENIED)
            self.assertEqual(denial.code, "permission_denied")

    def test_read_only_client_is_permitted_for_reads(self) -> None:
        self.assertIsNone(
            decide_agent_permission(
                AgentToolPermissionClass.READ, AgentClientPolicy.READ_ONLY
            )
        )

    def test_none_policy_denies_everything(self) -> None:
        for permission_class in AgentToolPermissionClass:
            denial = decide_agent_permission(permission_class, AgentClientPolicy.NONE)
            self.assertEqual(denial.outcome, AgentToolOutcome.PERMISSION_DENIED)

    def test_unknown_client_fails_closed(self) -> None:
        for permission_class in AgentToolPermissionClass:
            denial = decide_agent_permission(permission_class, None)
            self.assertEqual(denial.outcome, AgentToolOutcome.UNKNOWN_CLIENT)
            self.assertEqual(denial.code, "unknown_client")

    def test_live_writes_require_a_resolved_operation(self) -> None:
        with self.assertRaises(AgentPermissionValidationError):
            decide_agent_permission(
                AgentToolPermissionClass.LIVE_WRITE, AgentClientPolicy.FULL
            )

    def test_no_write_is_execution_ready_today(self) -> None:
        for operation in WriteOperation:
            denial = decide_agent_permission(
                AgentToolPermissionClass.LIVE_WRITE,
                AgentClientPolicy.FULL,
                live_write_operation=operation,
            )
            self.assertEqual(denial.outcome, AgentToolOutcome.NOT_EXECUTION_READY)
            self.assertEqual(denial.code, "not_execution_ready")
            self.assertIn("is not execution-ready", denial.message)

    def test_unknown_operation_in_the_matrix_fails_closed(self) -> None:
        denial = decide_agent_permission(
            AgentToolPermissionClass.LIVE_WRITE,
            AgentClientPolicy.FULL,
            live_write_operation=TRACK,
            capability_matrix={},
        )
        self.assertEqual(denial.outcome, AgentToolOutcome.NOT_EXECUTION_READY)

    def test_an_execution_ready_capability_is_permitted(self) -> None:
        real = WRITE_CAPABILITY_MATRIX[TRACK]
        ready = WriteCapability(
            operation=TRACK,
            entity_type=real.entity_type,
            field_path=real.field_path,
            domain_permission=DomainPermission.ALLOWED,
            capability_verified=True,
            adapter_implemented=True,
            readback_verified=True,
            readback_implemented=True,
            readback_strategy=real.readback_strategy,
            readback_field_path=real.readback_field_path,
        )
        self.assertIsNone(
            decide_agent_permission(
                AgentToolPermissionClass.LIVE_WRITE,
                AgentClientPolicy.FULL,
                live_write_operation=TRACK,
                capability_matrix={TRACK: ready},
            )
        )

    def test_denial_is_a_stable_typed_refusal(self) -> None:
        denial = decide_agent_permission(
            AgentToolPermissionClass.MUTATE, AgentClientPolicy.READ_ONLY
        )
        self.assertEqual(
            denial,
            PermissionDenial(
                AgentToolOutcome.PERMISSION_DENIED,
                "permission_denied",
                denial.message,
            ),
        )


class WriteCapabilitySummaryTest(unittest.TestCase):
    def test_summary_projects_every_sealed_operation(self) -> None:
        summary = write_capability_summary()
        self.assertEqual(len(summary), len(WRITE_CAPABILITY_MATRIX))
        by_operation = {entry["operation"]: entry for entry in summary}
        for operation in WriteOperation:
            self.assertIn(operation.value, by_operation)
            entry = by_operation[operation.value]
            self.assertIn("execution_ready", entry)
            self.assertFalse(entry["execution_ready"])  # sealed fact: nothing is ready today
            self.assertIn("domain_permission", entry)
            self.assertIn("capability_verified", entry)
            self.assertIn("adapter_implemented", entry)
            self.assertIn("readback_implemented", entry)


if __name__ == "__main__":
    unittest.main()
