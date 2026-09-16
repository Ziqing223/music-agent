"""M1: the frozen MCP tool surface -- a strict projection of the provider schemas.

These tests pin the verified contract arithmetic (15 READ + 5 MUTATE + 9
NEEDS_WRAPPER = 29 MCP tools), the provider-order projection, verbatim
description / deep-copied input-schema preservation, the hard exclusion of the
LIVE_WRITE + CLI-only tools, and the fail-closed drift detector
(MUSIC_AGENT_MCP_CONTRACT_v0.2.1 §7/§8.1/§17).
"""

from __future__ import annotations

import unittest
from copy import deepcopy
from unittest import mock

from music_agent.agent_tools import AGENT_TOOL_REGISTRY, AgentToolPermissionClass
from music_agent.mcp_projection import (
    DO_NOT_EXPOSE_TOOL_NAMES,
    MCP_EXPOSABLE_TOOL_NAMES,
    MUTATE_TOOL_NAMES,
    READ_TOOL_NAMES,
    WRAPPED_TOOL_NAMES,
    McpProjectionError,
    build_mcp_tool_definitions,
)
from music_agent.provider_tools import PROVIDER_TOOL_SCHEMAS

PROVIDER_NAMES = {schema.name for schema in PROVIDER_TOOL_SCHEMAS}
LIVE_WRITE_TWO = {"execute_write_intent", "add_catalog_to_library"}


class SurfaceArithmeticTest(unittest.TestCase):
    def test_frozen_group_counts(self) -> None:
        self.assertEqual(len(READ_TOOL_NAMES), 15)
        self.assertEqual(len(MUTATE_TOOL_NAMES), 5)
        self.assertEqual(len(WRAPPED_TOOL_NAMES), 9)
        self.assertEqual(len(MCP_EXPOSABLE_TOOL_NAMES), 29)

    def test_groups_are_disjoint(self) -> None:
        self.assertFalse(READ_TOOL_NAMES & MUTATE_TOOL_NAMES)
        self.assertFalse(READ_TOOL_NAMES & WRAPPED_TOOL_NAMES)
        self.assertFalse(MUTATE_TOOL_NAMES & WRAPPED_TOOL_NAMES)
        self.assertEqual(
            MCP_EXPOSABLE_TOOL_NAMES,
            READ_TOOL_NAMES | MUTATE_TOOL_NAMES | WRAPPED_TOOL_NAMES,
        )

    def test_do_not_expose_is_hard_walled_off(self) -> None:
        self.assertEqual(
            DO_NOT_EXPOSE_TOOL_NAMES,
            {"execute_write_intent", "add_catalog_to_library", "advance_preview"},
        )
        self.assertFalse(MCP_EXPOSABLE_TOOL_NAMES & DO_NOT_EXPOSE_TOOL_NAMES)

    def test_every_exposable_name_is_registered(self) -> None:
        for name in sorted(MCP_EXPOSABLE_TOOL_NAMES):
            self.assertIsNotNone(AGENT_TOOL_REGISTRY.lookup(name), name)

    def test_read_group_is_permission_class_read(self) -> None:
        for name in sorted(READ_TOOL_NAMES):
            spec = AGENT_TOOL_REGISTRY.lookup(name)
            self.assertIs(spec.permission_class, AgentToolPermissionClass.READ, name)

    def test_mutate_and_wrapped_groups_are_permission_class_mutate(self) -> None:
        for name in sorted(MUTATE_TOOL_NAMES | WRAPPED_TOOL_NAMES):
            spec = AGENT_TOOL_REGISTRY.lookup(name)
            self.assertIs(spec.permission_class, AgentToolPermissionClass.MUTATE, name)

    def test_repo_fact_live_write_schemas_exist_and_must_be_filtered(self) -> None:
        """Repo truth: the two LIVE_WRITE tools ARE in PROVIDER_TOOL_SCHEMAS (the
        reason §17.1 makes the projection filter explicit) and advance_preview
        is absent from them by design."""
        self.assertTrue(LIVE_WRITE_TWO <= PROVIDER_NAMES)
        self.assertNotIn("advance_preview", PROVIDER_NAMES)
        self.assertFalse(LIVE_WRITE_TWO & MCP_EXPOSABLE_TOOL_NAMES)


class ProjectionTest(unittest.TestCase):
    def test_twenty_nine_definition_shape(self) -> None:
        definitions = build_mcp_tool_definitions()
        self.assertEqual(len(definitions), 29)
        for definition in definitions:
            with self.subTest(tool=definition["name"]):
                self.assertEqual(
                    set(definition),
                    {"name", "description", "inputSchema"},
                )
                self.assertIn(definition["name"], MCP_EXPOSABLE_TOOL_NAMES)
                self.assertEqual(definition["inputSchema"]["type"], "object")
                self.assertIs(
                    definition["inputSchema"]["additionalProperties"], False
                )

    def test_definitions_follow_provider_order(self) -> None:
        definitions = build_mcp_tool_definitions()
        expected = [s.name for s in PROVIDER_TOOL_SCHEMAS if s.name in MCP_EXPOSABLE_TOOL_NAMES]
        self.assertEqual([d["name"] for d in definitions], expected)

    def test_definitions_carry_verbatim_provider_descriptions(self) -> None:
        by_name = {s.name: s for s in PROVIDER_TOOL_SCHEMAS}
        for definition in build_mcp_tool_definitions():
            self.assertEqual(
                definition["description"], by_name[definition["name"]].description
            )

    def test_definitions_are_deep_copies(self) -> None:
        first = build_mcp_tool_definitions()
        first[0]["inputSchema"]["properties"] = {"mutated": {}}
        first[0]["description"] = "mutated"
        second = build_mcp_tool_definitions()
        self.assertNotIn("mutated", second[0]["inputSchema"])
        self.assertNotEqual(second[0]["description"], "mutated")

    def test_definitions_never_include_do_not_expose(self) -> None:
        projected = {d["name"] for d in build_mcp_tool_definitions()}
        self.assertFalse(projected & DO_NOT_EXPOSE_TOOL_NAMES)
        self.assertEqual(projected, MCP_EXPOSABLE_TOOL_NAMES)

    def test_missing_provider_schema_fails_closed(self) -> None:
        """Drift detector: drop one allowlisted schema from the provider
        table and the projection must refuse to build a surface."""
        from music_agent import mcp_projection

        pruned = [s for s in PROVIDER_TOOL_SCHEMAS if s.name != "get_agent_capabilities"]
        with mock.patch.object(mcp_projection, "PROVIDER_TOOL_SCHEMAS", pruned):
            with self.assertRaises(McpProjectionError) as context:
                build_mcp_tool_definitions()
        self.assertIn("get_agent_capabilities", str(context.exception))

    def test_stray_provider_schema_fails_closed(self) -> None:
        """Drift detector: a new unfiltered provider schema (not in the frozen
        surface and not a known LIVE_WRITE filter) must refuse the build."""
        from music_agent import mcp_projection

        inflated = list(PROVIDER_TOOL_SCHEMAS) + [
            mock.Mock(
                name="some_future_read_tool",
                description="future",
                input_schema={"type": "object", "additionalProperties": False},
            )
        ]
        with mock.patch.object(mcp_projection, "PROVIDER_TOOL_SCHEMAS", inflated):
            with self.assertRaises(McpProjectionError) as context:
                build_mcp_tool_definitions()
        self.assertIn("some_future_read_tool", str(context.exception))


if __name__ == "__main__":
    unittest.main()