"""P09.2: the agent tool registry -- stable names, permission classes, fail-closed payload shapes.

These tests prove the frozen callable surface model clients invoke: every tool is registered
under its stable name with the correct permission class, valid payload envelopes pass, and
malformed envelopes -- unknown keys, missing keys, wrong types, foreign ID namespaces, naive
timestamps, boolean-where-integer -- fail closed with the shared validation error. The registry
itself refuses inconsistent registrations and returns ``None`` (never a guess) for unknown tools.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from music_agent.agent_tools import (
    AGENT_TOOL_REGISTRY,
    AgentToolName,
    AgentToolPermissionClass,
    AgentToolRegistry,
    AgentToolSpec,
    AgentToolValidationError,
)

TRACK_ID = "trk_11111111-1111-4111-8111-111111111111"
RUN_ID = "rcm_22222222-2222-4222-8222-222222222222"
CANDIDATE_ID = "cnd_33333333-3333-4333-8333-333333333333"
FEEDBACK_ID = "fbk_44444444-4444-4444-8444-444444444444"
INTENT_ID = "int_55555555-5555-4555-8555-555555555555"
ISO = "2026-08-16T00:00:00+00:00"
NOW = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)


def _valid_payloads() -> list[tuple[AgentToolName, dict]]:
    return [
        (AgentToolName.GET_CANONICAL_ENTITY, {"canonical_id": TRACK_ID}),
        (AgentToolName.QUERY_TRACK_PREFERENCE, {"target_id": TRACK_ID}),
        (
            AgentToolName.QUERY_TRACK_PREFERENCE,
            {"target_id": TRACK_ID, "source_system": "apple_music"},
        ),
        (AgentToolName.LIST_RECOMMENDATION_RUNS, {}),
        (AgentToolName.GET_RECOMMENDATION_RUN, {"run_id": RUN_ID}),
        (
            AgentToolName.GENERATE_RECOMMENDATION,
            {
                "target_ids": [TRACK_ID],
                "limit": 5,
                "source_system": "apple_music",
            },
        ),
        (
            AgentToolName.GENERATE_INFERRED_RECOMMENDATION,
            {"target_ids": [TRACK_ID], "limit": 5},
        ),
        (
            # P15-S3-S3C: inferred-only optional exploration floor, 0 <= v <= limit.
            AgentToolName.GENERATE_INFERRED_RECOMMENDATION,
            {"target_ids": [TRACK_ID], "limit": 5, "min_exploration": 1},
        ),
        (AgentToolName.LIST_FEEDBACK_OBSERVATIONS, {}),
        (AgentToolName.GET_FEEDBACK_OBSERVATION, {"feedback_id": FEEDBACK_ID}),
        (
            AgentToolName.RECORD_FEEDBACK,
            {
                "kind": "liked",
                "source_system": "recommendation_ui",
                "source_path": "card_actions",
                "target_id": TRACK_ID,
                "run_id": None,
                "candidate_id": None,
                "attribution": None,
                "event_at": None,
                "source_event_id": None,
                "feedback_id": None,
            },
        ),
        (AgentToolName.INTERPRET_FEEDBACK, {"feedback_id": FEEDBACK_ID}),
        (AgentToolName.LIST_LEARNING_APPLICATIONS, {}),
        (AgentToolName.GET_LEARNING_APPLICATION, {"feedback_id": FEEDBACK_ID}),
        (AgentToolName.APPLY_LEARNING, {"feedback_id": FEEDBACK_ID}),
        (AgentToolName.GET_AGENT_CAPABILITIES, {}),
        (AgentToolName.EXECUTE_WRITE_INTENT, {"intent_id": INTENT_ID}),
        (AgentToolName.DISCOVER_CATALOG_TRACKS, {"term": "midnight city"}),
        (AgentToolName.DISCOVER_CATALOG_TRACKS, {"term": "midnight city", "limit": 5}),
        # P15-S3-S2: durable catalog-track memory read -- canonical_id XOR term.
        (
            AgentToolName.QUERY_CATALOG_DISCOVERY_STATE,
            {"canonical_id": TRACK_ID},
        ),
        (AgentToolName.QUERY_CATALOG_DISCOVERY_STATE, {"term": "J-Pop"}),
        (
            AgentToolName.QUERY_CATALOG_DISCOVERY_STATE,
            {"term": "J-Pop", "limit": 5},
        ),
    ]


class ToolRegistryTest(unittest.TestCase):
    def test_every_tool_is_registered_under_its_stable_name(self) -> None:
        self.assertEqual(set(AGENT_TOOL_REGISTRY.tool_names), set(name.value for name in AgentToolName))
        for name in AgentToolName:
            spec = AGENT_TOOL_REGISTRY.lookup(name)
            self.assertIsNotNone(spec)
            self.assertEqual(spec.name, name)

    def test_unknown_tool_lookup_returns_none(self) -> None:
        self.assertIsNone(AGENT_TOOL_REGISTRY.lookup("not_a_tool"))
        self.assertIsNone(AGENT_TOOL_REGISTRY.lookup(""))
        self.assertIsNone(AGENT_TOOL_REGISTRY.lookup(7))  # type: ignore[arg-type]

    def test_permission_classes_match_the_planned_surface(self) -> None:
        read_tools = {
            AgentToolName.GET_CANONICAL_ENTITY,
            AgentToolName.QUERY_TRACK_PREFERENCE,
            AgentToolName.LIST_RECOMMENDATION_RUNS,
            AgentToolName.GET_RECOMMENDATION_RUN,
            AgentToolName.LIST_FEEDBACK_OBSERVATIONS,
            AgentToolName.GET_FEEDBACK_OBSERVATION,
            AgentToolName.INTERPRET_FEEDBACK,
            AgentToolName.LIST_LEARNING_APPLICATIONS,
            AgentToolName.GET_LEARNING_APPLICATION,
            AgentToolName.GET_AGENT_CAPABILITIES,
            AgentToolName.GET_NOW_PLAYING,
            AgentToolName.GET_ACTIVE_CONTEXT,  # P14-C06.2: pure read of the in-memory register
            # P14-R3.1: store-only name lookup, zero writes.
            AgentToolName.SEARCH_LIBRARY_TRACKS,
            # P15-PC: playback-continuity observation is a pure read of the in-memory
            # registers plus on-demand adapter/runner reads.
            AgentToolName.GET_PLAYBACK_CONTEXT,
            # P15-S3-S2: durable catalog-track memory read (facts + derived labels only).
            AgentToolName.QUERY_CATALOG_DISCOVERY_STATE,
        }
        mutate_tools = {
            AgentToolName.GENERATE_RECOMMENDATION,
            AgentToolName.RECORD_FEEDBACK,
            AgentToolName.APPLY_LEARNING,
            # P10.12: transient playback is client-policy-gated (MUTATE), deliberately
            # NOT LIVE_WRITE -- the sealed library-write capability matrix is untouched.
            AgentToolName.PLAY,
            AgentToolName.PAUSE,
            AgentToolName.NEXT_TRACK,
            AgentToolName.PREVIOUS_TRACK,
            AgentToolName.PLAY_TRACK,
            AgentToolName.GENERATE_INFERRED_RECOMMENDATION,
            # P11.3: preview is transient (like playback), never a library mutation.
            AgentToolName.PREVIEW_CATALOG_TRACK,
            # P3B batch 2: stopping the agent's own preview is transient, like playback.
            AgentToolName.STOP_PREVIEW,
            # P11-T1: catalog discovery writes durable staging candidates.
            AgentToolName.DISCOVER_CATALOG_TRACKS,
            # P15-S1: a continuous preview session is transient audio (like playback).
            AgentToolName.PREVIEW_BATCH,
            # P15-S1 C02: advancing the running session is transient audio routing.
            AgentToolName.ADVANCE_PREVIEW,
            # P16-S4: opening Apple's official track page is a browser side effect,
            # never a library/playback mutation (MUTATE, like preview).
            AgentToolName.OPEN_IN_APPLE_MUSIC,
        }
        live_write_tools = {
            AgentToolName.EXECUTE_WRITE_INTENT,
            # P11.3: library add is a real external mutation, capability-gated.
            AgentToolName.ADD_CATALOG_TO_LIBRARY,
        }
        for name in AgentToolName:
            spec = AGENT_TOOL_REGISTRY.lookup(name)
            if name in read_tools:
                self.assertEqual(spec.permission_class, AgentToolPermissionClass.READ, name)
            elif name in mutate_tools:
                self.assertEqual(spec.permission_class, AgentToolPermissionClass.MUTATE, name)
            else:
                self.assertIn(name, live_write_tools)
                self.assertEqual(spec.permission_class, AgentToolPermissionClass.LIVE_WRITE, name)

    def test_registry_rejects_inconsistent_registrations(self) -> None:
        with self.assertRaises(AgentToolValidationError):
            AgentToolRegistry(
                {"get_canonical_entity": AgentToolSpec(
                    AgentToolName.QUERY_TRACK_PREFERENCE,
                    AgentToolPermissionClass.READ,
                    lambda payload: None,
                )}
            )
        with self.assertRaises(AgentToolValidationError):
            AgentToolRegistry({"" : AgentToolSpec(  # noqa: E203
                AgentToolName.GET_CANONICAL_ENTITY,
                AgentToolPermissionClass.READ,
                lambda payload: None,
            )})


def _record_feedback_base() -> dict:
    for name, payload in _valid_payloads():
        if name is AgentToolName.RECORD_FEEDBACK:
            return payload
    raise AssertionError("record_feedback payload missing")


class PayloadValidationTest(unittest.TestCase):
    def test_valid_payloads_pass(self) -> None:
        for name, payload in _valid_payloads():
            with self.subTest(tool=name.value):
                AGENT_TOOL_REGISTRY.lookup(name).validate(payload)

    def test_empty_object_tools_reject_extra_keys(self) -> None:
        for name in (
            AgentToolName.LIST_RECOMMENDATION_RUNS,
            AgentToolName.LIST_FEEDBACK_OBSERVATIONS,
            AgentToolName.LIST_LEARNING_APPLICATIONS,
            AgentToolName.GET_AGENT_CAPABILITIES,
        ):
            with self.subTest(tool=name.value):
                with self.assertRaises(AgentToolValidationError):
                    AGENT_TOOL_REGISTRY.lookup(name).validate({"extra": True})

    def test_foreign_id_namespaces_fail_closed(self) -> None:
        cases = (
            (AgentToolName.QUERY_TRACK_PREFERENCE, {"target_id": RUN_ID}),
            (AgentToolName.GET_RECOMMENDATION_RUN, {"run_id": FEEDBACK_ID}),
            (AgentToolName.GET_FEEDBACK_OBSERVATION, {"feedback_id": RUN_ID}),
            (AgentToolName.EXECUTE_WRITE_INTENT, {"intent_id": RUN_ID}),
        )
        for name, payload in cases:
            with self.subTest(tool=name.value):
                with self.assertRaises(AgentToolValidationError):
                    AGENT_TOOL_REGISTRY.lookup(name).validate(payload)

    def test_model_timestamps_fail_closed(self) -> None:
        """P16-S1: ``observed_at`` / ``applied_at`` are NOT part of the
        feedback/learning envelopes. The durable times are service-authoritative
        (the trusted ``completed_at`` execution instant, never model input); a
        model that sends either key anyway is rejected as an unknown key --
        fail-closed even for well-formed timestamps."""
        with self.assertRaises(AgentToolValidationError):
            AGENT_TOOL_REGISTRY.lookup(AgentToolName.RECORD_FEEDBACK).validate(
                {**_record_feedback_base(), "observed_at": "2026-08-16T00:00:00"}
            )
        with self.assertRaises(AgentToolValidationError):
            AGENT_TOOL_REGISTRY.lookup(AgentToolName.RECORD_FEEDBACK).validate(
                {**_record_feedback_base(), "observed_at": "2026-08-16T00:00:00+00:00"}
            )
        with self.assertRaises(AgentToolValidationError):
            AGENT_TOOL_REGISTRY.lookup(AgentToolName.APPLY_LEARNING).validate(
                {"feedback_id": FEEDBACK_ID, "applied_at": ISO}
            )
        with self.assertRaises(AgentToolValidationError):
            AGENT_TOOL_REGISTRY.lookup(AgentToolName.GENERATE_RECOMMENDATION).validate(
                {"target_ids": [TRACK_ID], "limit": 5, "produced_at": "2026-08-16T00:00:00"}
            )  # P15 burn-down Issue 1: produced_at is an unknown key now (authority-owned).

    def test_generate_tools_reject_produced_at_as_unknown_key(self) -> None:
        """P15 burn-down Issue 1: ``produced_at`` is NOT part of either generation
        envelope. The durable run time is service-authoritative (the trusted
        ``completed_at`` execution instant, never model input); a model that
        sends the key anyway is rejected as an unknown key -- fail-closed even
        for well-formed timestamps."""
        for tool in (
            AgentToolName.GENERATE_RECOMMENDATION,
            AgentToolName.GENERATE_INFERRED_RECOMMENDATION,
        ):
            with self.subTest(tool=tool.value):
                with self.assertRaises(AgentToolValidationError) as raised:
                    AGENT_TOOL_REGISTRY.lookup(tool).validate(
                        {
                            "target_ids": [TRACK_ID],
                            "limit": 5,
                            "produced_at": "2026-08-16T10:00:00+00:00",
                        }
                    )
                self.assertIn("payload keys must be one of", str(raised.exception))

    def test_discover_catalog_tracks_shape_rules(self) -> None:
        spec = AGENT_TOOL_REGISTRY.lookup(AgentToolName.DISCOVER_CATALOG_TRACKS)
        for payload in (
            {},
            {"term": ""},
            {"term": "x", "limit": 0},
            {"term": "x", "limit": True},
            {"term": "x", "extra": 1},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(AgentToolValidationError):
                    spec.validate(payload)

    def test_search_library_tracks_shape_rules(self) -> None:
        spec = AGENT_TOOL_REGISTRY.lookup(AgentToolName.SEARCH_LIBRARY_TRACKS)
        for payload in (
            {},
            {"term": ""},
            {"term": "x", "limit": 0},
            {"term": "x", "limit": True},
            {"term": "x", "limit": "5"},
            {"term": "x", "extra": 1},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(AgentToolValidationError):
                    spec.validate(payload)
        spec.validate({"term": "初行", "limit": 5})

    def test_query_catalog_discovery_state_shape_rules(self) -> None:
        spec = AGENT_TOOL_REGISTRY.lookup(AgentToolName.QUERY_CATALOG_DISCOVERY_STATE)
        for payload in (
            {},
            {"canonical_id": None},
            {"term": None},
            {"canonical_id": TRACK_ID, "term": "J-Pop"},  # not XOR
            {"canonical_id": RUN_ID},  # foreign id namespace
            {"term": ""},
            {"term": 7},
            {"term": "J-Pop", "limit": 0},
            {"term": "J-Pop", "limit": True},
            {"term": "J-Pop", "limit": "5"},
            {"term": "J-Pop", "extra": 1},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(AgentToolValidationError):
                    spec.validate(payload)
        spec.validate({"canonical_id": TRACK_ID})
        spec.validate({"term": "初行极光"})
        spec.validate({"term": "J-Pop", "limit": 50})

    def test_generate_recommendation_shape_rules(self) -> None:
        spec = AGENT_TOOL_REGISTRY.lookup(AgentToolName.GENERATE_RECOMMENDATION)
        for payload in (
            {"target_ids": [], "limit": 5},
            {"target_ids": [RUN_ID], "limit": 5},
            {"target_ids": [TRACK_ID], "limit": 0},
            {"target_ids": [TRACK_ID], "limit": True},
            {"target_ids": [TRACK_ID], "limit": 5, "extra": 1},
            {"target_ids": [TRACK_ID], "limit": 5, "exclude_target_ids": []},
            {"target_ids": [TRACK_ID], "limit": 5, "exclude_target_ids": ["nope"]},
            {"target_ids": [TRACK_ID], "limit": 5, "avoid_previous_runs": "yes"},
            {"target_ids": [TRACK_ID], "limit": 5, "genres": []},
            {"target_ids": [TRACK_ID], "limit": 5, "genres": ["  "]},
            {"target_ids": [TRACK_ID], "limit": 5, "genres": [7]},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(AgentToolValidationError):
                    spec.validate(payload)

    def test_generate_recommendation_accepts_exclusion_and_direction_args(self) -> None:
        spec = AGENT_TOOL_REGISTRY.lookup(AgentToolName.GENERATE_RECOMMENDATION)
        spec.validate(
            {
                "target_ids": [TRACK_ID],
                "limit": 5,
                "exclude_target_ids": [TRACK_ID],
                "avoid_previous_runs": True,
                "genres": ["J-Pop"],
            }
        )

    def test_generate_inferred_rejects_min_exploration_over_small_limit(self) -> None:
        spec = AGENT_TOOL_REGISTRY.lookup(AgentToolName.GENERATE_INFERRED_RECOMMENDATION)
        with self.assertRaises(AgentToolValidationError):
            spec.validate(
                {"target_ids": [TRACK_ID], "limit": 2, "min_exploration": 3}
            )

    def test_record_feedback_target_or_recommendation_rule(self) -> None:
        spec = AGENT_TOOL_REGISTRY.lookup(AgentToolName.RECORD_FEEDBACK)
        base = _record_feedback_base()
        with self.assertRaises(AgentToolValidationError):
            spec.validate({**base, "target_id": None})  # neither target nor recommendation
        with self.assertRaises(AgentToolValidationError):
            spec.validate({**base, "run_id": RUN_ID})  # candidate_id missing
        with self.assertRaises(AgentToolValidationError):
            # both target and recommendation present
            spec.validate({**base, "run_id": RUN_ID, "candidate_id": CANDIDATE_ID})
        spec.validate({**base, "target_id": None, "run_id": RUN_ID, "candidate_id": CANDIDATE_ID})

    def test_record_feedback_rejects_unknown_kind_and_bad_attribution(self) -> None:
        spec = AGENT_TOOL_REGISTRY.lookup(AgentToolName.RECORD_FEEDBACK)
        base = _record_feedback_base()
        with self.assertRaises(AgentToolValidationError):
            spec.validate({**base, "kind": "exploded"})
        with self.assertRaises(AgentToolValidationError):
            spec.validate(
                {
                    **base,
                    "kind": "attribution_correction",
                    "attribution": {"aspect_kind": "mystery", "aspect_id": "x", "relation": "excluded"},
                }
            )
        spec.validate(
            {
                **base,
                "kind": "attribution_correction",
                "attribution": {
                    "aspect_kind": "artist",
                    "aspect_id": "art_11111111-1111-4111-8111-111111111111",
                    "relation": "excluded",
                },
            }
        )


class GenerateInferredExplorationFloorValidationTest(unittest.TestCase):
    """P15-S3-S3C contract items: the inferred-only optional ``min_exploration`` gate.

    Covers acceptance within ``[0, limit]``, the default-0 omission contract, the
    fail-closed rejections (negative, over-limit, non-integer), the plain tool's
    strict-envelope rejection of the key, and the over-small-limit bound pairing.
    """

    def _inferred(self) -> AgentToolSpec:
        return AGENT_TOOL_REGISTRY.lookup(AgentToolName.GENERATE_INFERRED_RECOMMENDATION)

    def _base(self, **extra) -> dict:
        payload = {"target_ids": [TRACK_ID], "limit": 5}
        payload.update(extra)
        return payload

    def test_inferred_accepts_min_exploration_within_bounds(self) -> None:
        spec = self._inferred()
        for value in (0, 1, 3, 5):
            with self.subTest(value=value):
                spec.validate(self._base(min_exploration=value))

    def test_inferred_omission_keeps_ordinary_recommendation(self) -> None:
        # A payload without the key has always validated and still does -- the
        # floor defaults to 0 and the ordinary recommendation path is untouched.
        self._inferred().validate(self._base())

    def test_inferred_null_is_treated_as_absence(self) -> None:
        # Explicit null matches the codebase's optional-arg convention: absent.
        self._inferred().validate(self._base(min_exploration=None))

    def test_inferred_rejects_negative_and_over_limit(self) -> None:
        spec = self._inferred()
        for value in (-1, 6, 100):
            with self.subTest(value=value):
                with self.assertRaises(AgentToolValidationError):
                    spec.validate(self._base(min_exploration=value))

    def test_inferred_rejects_non_integer_min_exploration(self) -> None:
        spec = self._inferred()
        for value in (True, False, 1.5, "1", [1]):
            with self.subTest(value=value):
                with self.assertRaises(AgentToolValidationError):
                    spec.validate(self._base(min_exploration=value))

    def test_plain_generate_rejects_min_exploration(self) -> None:
        spec = AGENT_TOOL_REGISTRY.lookup(AgentToolName.GENERATE_RECOMMENDATION)
        with self.assertRaises(AgentToolValidationError):
            spec.validate(
                {"target_ids": [TRACK_ID], "limit": 5, "min_exploration": 1}
            )


class GenerateInferredFreshFloorValidationTest(unittest.TestCase):
    """P15-S3-S3D contract items: the inferred-only optional ``min_fresh`` gate.

    Mirrors the S3-S3C ``min_exploration`` envelope: acceptance within
    ``[0, limit]``, the default-0 omission contract, fail-closed rejections
    (negative, over-limit, non-integer), the plain tool's strict-envelope
    rejection, and the relationship pin is deliberately NOT validated here --
    the service normalizes it (effective exploration = max of both knobs).
    """

    def _inferred(self) -> AgentToolSpec:
        return AGENT_TOOL_REGISTRY.lookup(AgentToolName.GENERATE_INFERRED_RECOMMENDATION)

    def _base(self, **extra) -> dict:
        payload = {"target_ids": [TRACK_ID], "limit": 5}
        payload.update(extra)
        return payload

    def test_inferred_accepts_min_fresh_within_bounds(self) -> None:
        spec = self._inferred()
        for value in (0, 1, 3, 5):
            with self.subTest(value=value):
                spec.validate(self._base(min_fresh=value))

    def test_inferred_omission_keeps_ordinary_recommendation(self) -> None:
        self._inferred().validate(self._base())

    def test_inferred_null_is_treated_as_absence(self) -> None:
        self._inferred().validate(self._base(min_fresh=None))

    def test_inferred_rejects_negative_and_over_limit(self) -> None:
        spec = self._inferred()
        for value in (-1, 6, 100):
            with self.subTest(value=value):
                with self.assertRaises(AgentToolValidationError):
                    spec.validate(self._base(min_fresh=value))

    def test_inferred_rejects_non_integer_min_fresh(self) -> None:
        spec = self._inferred()
        for value in (True, False, 1.5, "1", [1]):
            with self.subTest(value=value):
                with self.assertRaises(AgentToolValidationError):
                    spec.validate(self._base(min_fresh=value))

    def test_plain_generate_rejects_min_fresh(self) -> None:
        spec = AGENT_TOOL_REGISTRY.lookup(AgentToolName.GENERATE_RECOMMENDATION)
        with self.assertRaises(AgentToolValidationError):
            spec.validate(
                {"target_ids": [TRACK_ID], "limit": 5, "min_fresh": 1}
            )

    def test_inferred_accepts_both_floors_independently(self) -> None:
        # The relationship between the knobs is service-level normalization
        # (Option B: effective minimum = max(...)), never a validator coupling --
        # each knob validates only against its own [0, limit] bound here.
        self._inferred().validate(self._base(min_fresh=2, min_exploration=4))


if __name__ == "__main__":
    unittest.main()
