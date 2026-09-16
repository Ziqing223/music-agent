"""MCP M1: the frozen public tool surface for the local stdio MCP adapter.

The MCP surface is a strict projection of the existing provider-facing tool
schemas -- one existing ``AgentToolName`` per MCP tool, no second vocabulary,
no re-implementation of validation, permission, or execution. The frozen
groupings below carry the verified contract arithmetic
(MUSIC_AGENT_MCP_CONTRACT_v0.2.1 §7):

    15 READ + 5 agent-owned durable MUTATE + 9 NEEDS_WRAPPER = 29 MCP tools

Excluded by construction (contract §7.4):

* ``execute_write_intent`` / ``add_catalog_to_library`` -- LIVE_WRITE tools
  that ARE present in ``PROVIDER_TOOL_SCHEMAS`` (31 entries); the projection
  filters them explicitly so schema reuse can never smuggle them into MCP;
* ``advance_preview`` -- registry-only / CLI-routed, absent from the provider
  schemas by design and never reintroduced here.

The projection fails closed on drift: an allowlisted name missing from the
provider schemas, a provider schema outside the frozen surface, a grouping
overlap, or a registry permission-class mismatch raises
:class:`McpProjectionError` at construction time instead of emitting a surface
nobody reviewed.

Projected definitions preserve the provider descriptions verbatim (Chinese,
the semantic carrier of the behavioral contract -- §8.1) and deep-copy the
input schema (``additionalProperties: false`` / enums / required fields
intact -- §17).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from music_agent.agent_tools import AGENT_TOOL_REGISTRY, AgentToolPermissionClass
from music_agent.provider_tools import PROVIDER_TOOL_SCHEMAS


class McpProjectionError(ValueError):
    code = "mcp_projection_error"


READ_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "get_canonical_entity",
        "query_track_preference",
        "list_recommendation_runs",
        "get_recommendation_run",
        "list_feedback_observations",
        "get_feedback_observation",
        "interpret_feedback",
        "list_learning_applications",
        "get_learning_application",
        "get_agent_capabilities",
        "get_now_playing",
        "get_active_context",
        "get_playback_context",
        "search_library_tracks",
        "query_catalog_discovery_state",
    }
)

MUTATE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "generate_recommendation",
        "record_feedback",
        "apply_learning",
        "generate_inferred_recommendation",
        "discover_catalog_tracks",
    }
)

WRAPPED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "play",
        "pause",
        "next_track",
        "previous_track",
        "play_track",
        "preview_catalog_track",
        "preview_batch",
        "stop_preview",
        "open_in_apple_music",
    }
)

DO_NOT_EXPOSE_TOOL_NAMES: frozenset[str] = frozenset(
    {"execute_write_intent", "add_catalog_to_library", "advance_preview"}
)

MCP_EXPOSABLE_TOOL_NAMES: frozenset[str] = (
    READ_TOOL_NAMES | MUTATE_TOOL_NAMES | WRAPPED_TOOL_NAMES
)

_EXPECTED_COUNTS = {
    "read": 15,
    "mutate": 5,
    "wrapped": 9,
}
_TOTAL_EXPOSABLE = sum(_EXPECTED_COUNTS.values())

# The two provider schemas the projection always drops (LIVE_WRITE; §17.1).
_LIVE_WRITE_PROVIDER_NAMES = frozenset({"execute_write_intent", "add_catalog_to_library"})


def _verify_surface() -> None:
    """Fail closed on any drift between the frozen surface and the repo facts."""
    if len(READ_TOOL_NAMES) != _EXPECTED_COUNTS["read"]:
        raise McpProjectionError(
            f"READ group must hold {_EXPECTED_COUNTS['read']} names, has {len(READ_TOOL_NAMES)}"
        )
    if len(MUTATE_TOOL_NAMES) != _EXPECTED_COUNTS["mutate"]:
        raise McpProjectionError(
            f"MUTATE group must hold {_EXPECTED_COUNTS['mutate']} names, has {len(MUTATE_TOOL_NAMES)}"
        )
    if len(WRAPPED_TOOL_NAMES) != _EXPECTED_COUNTS["wrapped"]:
        raise McpProjectionError(
            f"WRAPPED group must hold {_EXPECTED_COUNTS['wrapped']} names, has {len(WRAPPED_TOOL_NAMES)}"
        )
    groups = [READ_TOOL_NAMES, MUTATE_TOOL_NAMES, WRAPPED_TOOL_NAMES]
    for index, group in enumerate(groups):
        for other in groups[index + 1 :]:
            overlap = group & other
            if overlap:
                raise McpProjectionError(f"surface groups overlap: {sorted(overlap)}")
    if len(MCP_EXPOSABLE_TOOL_NAMES) != _TOTAL_EXPOSABLE:
        raise McpProjectionError(
            f"exportable surface must hold {_TOTAL_EXPOSABLE} names, has {len(MCP_EXPOSABLE_TOOL_NAMES)}"
        )
    overlap = MCP_EXPOSABLE_TOOL_NAMES & DO_NOT_EXPOSE_TOOL_NAMES
    if overlap:
        raise McpProjectionError(f"surface overlaps DO_NOT_EXPOSE: {sorted(overlap)}")
    provider_names = {schema.name for schema in PROVIDER_TOOL_SCHEMAS}
    missing = set(MCP_EXPOSABLE_TOOL_NAMES) - provider_names
    if missing:
        raise McpProjectionError(
            f"allowlisted tools missing from PROVIDER_TOOL_SCHEMAS: {sorted(missing)}"
        )
    stray = provider_names - set(MCP_EXPOSABLE_TOOL_NAMES) - _LIVE_WRITE_PROVIDER_NAMES
    if stray:
        raise McpProjectionError(
            "provider schemas outside the frozen surface (add to the surface "
            f"deliberately or filter them): {sorted(stray)}"
        )
    for name in sorted(MCP_EXPOSABLE_TOOL_NAMES):
        spec = AGENT_TOOL_REGISTRY.lookup(name)
        if spec is None:
            raise McpProjectionError(f"{name} is allowlisted but not registered")
        expected_class = (
            AgentToolPermissionClass.READ
            if name in READ_TOOL_NAMES
            else AgentToolPermissionClass.MUTATE
        )
        if spec.permission_class is not expected_class:
            raise McpProjectionError(
                f"{name} is registered as {spec.permission_class.value}, "
                f"expected {expected_class.value}"
            )


def build_mcp_tool_definitions() -> tuple[dict[str, Any], ...]:
    """Project the frozen public surface to the MCP wire tool definitions.

    One call produces the stable ``tools/list`` payload in provider-schema
    order: name, the unchanged provider description, and a deep-copied input
    schema (mutating a returned definition never mutates the surface).
    """
    _verify_surface()
    definitions: list[dict[str, Any]] = []
    for schema in PROVIDER_TOOL_SCHEMAS:
        if schema.name not in MCP_EXPOSABLE_TOOL_NAMES:
            continue
        if schema.input_schema.get("type") != "object":
            raise McpProjectionError(
                f"tool {schema.name} input schema must declare type object"
            )
        if schema.input_schema.get("additionalProperties") is not False:
            raise McpProjectionError(
                f"tool {schema.name} input schema must declare additionalProperties false"
            )
        definitions.append(
            {
                "name": schema.name,
                "description": schema.description,
                "inputSchema": deepcopy(dict(schema.input_schema)),
            }
        )
    if len(definitions) != _TOTAL_EXPOSABLE:
        raise McpProjectionError(
            f"projected {len(definitions)} tools, expected {_TOTAL_EXPOSABLE}"
        )
    return tuple(definitions)