"""P12: model-facing playback tool contract pins.

The provider-facing tool descriptions are the only contract the models see for
playback routing. These tests pin the required phrases so a future edit cannot
silently drop the transport/resume-only and targeted-playback obligations.
They pin contract text only -- they do not assert anything about how any
provider model behaves with the tools.
"""

import unittest

from music_agent.provider_tools import PROVIDER_TOOLS_BY_NAME


def _description(name: str) -> str:
    return PROVIDER_TOOLS_BY_NAME[name].description


class PlayToolContractTest(unittest.TestCase):
    """play = transport/resume only; never target selection."""

    def test_play_is_resume_only_on_current_context(self) -> None:
        desc = _description("play")
        self.assertIn("恢复", desc)
        self.assertIn("当前曲目", desc)
        self.assertIn("当前播放上下文", desc)

    def test_play_does_not_select_a_track(self) -> None:
        self.assertIn("绝不选择", _description("play"))

    def test_play_never_substitutes_for_play_track(self) -> None:
        desc = _description("play")
        self.assertIn("play_track", desc)
        self.assertIn("绝不能代替", desc)

    def test_play_takes_no_track_argument(self) -> None:
        self.assertEqual(
            dict(PROVIDER_TOOLS_BY_NAME["play"].input_schema),
            {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        )


class PlayTrackToolContractTest(unittest.TestCase):
    """play_track = the required tool for explicit named-track playback."""

    def test_play_track_selects_a_specific_canonical_track(self) -> None:
        desc = _description("play_track")
        self.assertIn("选择并播放", desc)
        self.assertIn("canonical_id", desc)

    def test_play_track_is_required_once_canonical_id_exists(self) -> None:
        desc = _description("play_track")
        self.assertIn("必须使用本工具", desc)
        self.assertIn("工具结果中已给出 canonical_id", desc)

    def test_generic_play_is_not_a_substitute(self) -> None:
        self.assertIn("不能作为本工具的替代", _description("play_track"))


class GetNowPlayingToolContractTest(unittest.TestCase):
    """get_now_playing = snapshot read; not proof of post-command state."""

    def test_snapshot_only_without_position_or_duration(self) -> None:
        desc = _description("get_now_playing")
        self.assertIn("快照", desc)
        self.assertIn("不含播放位置", desc)

    def test_pre_action_state_does_not_prove_post_action_state(self) -> None:
        desc = _description("get_now_playing")
        self.assertIn("不能证明", desc)

    def test_post_action_verification_requires_a_new_read(self) -> None:
        self.assertIn("动作之后再次调用本工具", _description("get_now_playing"))


class ListRecommendationRunsToolContractTest(unittest.TestCase):
    """Bounded history projection: recent default slice, never the full history."""

    def test_schema_exposes_optional_bounded_limit(self) -> None:
        self.assertEqual(
            dict(PROVIDER_TOOLS_BY_NAME["list_recommendation_runs"].input_schema),
            {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "返回的最近记录条数（默认 5）",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        )

    def test_description_promises_recent_default_and_total(self) -> None:
        desc = _description("list_recommendation_runs")
        self.assertIn("默认只返回最近 5 条", desc)
        self.assertIn("runs_total 为历史总数", desc)


class GenerationArgsToolContractTest(unittest.TestCase):
    """Both generate tools expose exclusion + direction args so the model can express
    「换一组」and user directions without touching the scoring algorithm."""

    def test_generate_tools_expose_exclusion_and_direction_args(self) -> None:
        for name in ("generate_recommendation", "generate_inferred_recommendation"):
            with self.subTest(tool=name):
                properties = PROVIDER_TOOLS_BY_NAME[name].input_schema["properties"]
                self.assertIn("exclude_target_ids", properties)
                self.assertIn("avoid_previous_runs", properties)
                self.assertIn("genres", properties)
                for key in ("exclude_target_ids", "avoid_previous_runs", "genres"):
                    self.assertNotIn(
                        key, PROVIDER_TOOLS_BY_NAME[name].input_schema["required"]
                    )

    def test_generate_descriptions_still_require_positive_limit(self) -> None:
        for name in ("generate_recommendation", "generate_inferred_recommendation"):
            schema = PROVIDER_TOOLS_BY_NAME[name].input_schema
            self.assertIn("limit", schema["required"])
            self.assertGreaterEqual(schema["properties"]["limit"]["minimum"], 1)

    def test_generate_schemas_expose_no_produced_at(self) -> None:
        """P15 burn-down Issue 1: run time is not a model argument. Neither
        generation schema exposes ``produced_at`` -- the service owns the
        execution instant through its trusted completion context, so the model
        cannot even be offered the key it must never control."""
        for name in ("generate_recommendation", "generate_inferred_recommendation"):
            with self.subTest(tool=name):
                schema = PROVIDER_TOOLS_BY_NAME[name].input_schema
                self.assertNotIn("produced_at", schema["properties"])
                self.assertNotIn("produced_at", schema["required"])


class T15DefaultExclusionContractTest(unittest.TestCase):
    """P19-T15: the models that forget to pass exclusion arguments must still
    get recent-run dedup -- the server default is stated in both generate
    tools' descriptions AND in the avoid_previous_runs field description, so
    never passing the key cannot re-enable repeats silently."""

    def test_both_generate_descriptions_state_the_default_exclusion(self) -> None:
        for name in ("generate_recommendation", "generate_inferred_recommendation"):
            with self.subTest(tool=name):
                desc = _description(name)
                self.assertIn(
                    "调用未提供任何排除参数时本工具默认排除最近 5 批推荐过的轨道",
                    desc,
                )
                self.assertIn("显式传 avoid_previous_runs=false 才允许", desc)

    def test_avoid_previous_runs_field_states_the_default(self) -> None:
        for name in ("generate_recommendation", "generate_inferred_recommendation"):
            with self.subTest(tool=name):
                field = PROVIDER_TOOLS_BY_NAME[name].input_schema["properties"][
                    "avoid_previous_runs"
                ]["description"]
                self.assertIn("缺省 true", field)
                self.assertIn("显式 false 允许最近推荐过的轨道重新出现", field)


class GenerateToolEligibilityContractTest(unittest.TestCase):
    """P15-S4-M2-2: both generate descriptions state the eligibility reality --
    target_ids is a scope, not an arbitrary catalog injection, and the inferred
    tool is the successor for fresh catalog tracks without direct evidence."""

    def test_generate_recommendation_names_its_evidence_requirement(self) -> None:
        desc = _description("generate_recommendation")
        self.assertIn("target_ids 仅为目标范围（偏好引用），不是任意目录注入", desc)
        self.assertIn("方向性直接偏好证据", desc)
        # P16-S2: the deterministic fallback replaces the old switch-to-inferred
        # pointer; the fresh guidance stays explicit (min_fresh is unavailable
        # on the plain tool).
        self.assertIn("会自动改用推断通道生成同一批推荐", desc)
        self.assertIn("channel=inferred_fallback", desc)
        self.assertIn("min_fresh", desc)

    def test_generate_recommendation_target_ids_is_a_scope_not_a_pool(self) -> None:
        field = PROVIDER_TOOLS_BY_NAME["generate_recommendation"].input_schema[
            "properties"
        ]["target_ids"]["description"]
        self.assertIn("目标轨道 id 列表", field)
        self.assertIn("偏好引用", field)
        self.assertNotIn("候选轨道 id 列表", field)

    def test_generate_inferred_is_the_catalog_successor(self) -> None:
        desc = _description("generate_inferred_recommendation")
        self.assertIn(
            "catalog 发现之后对「无直接证据曲目」的常规推荐接续", desc
        )
        self.assertIn("generate_recommendation 无法为它们推荐", desc)
        self.assertIn("亲和度推断", desc)

    def test_generate_inferred_target_ids_admits_inferred_entries(self) -> None:
        field = PROVIDER_TOOLS_BY_NAME["generate_inferred_recommendation"].input_schema[
            "properties"
        ]["target_ids"]["description"]
        self.assertIn("目标轨道 id 列表", field)
        self.assertIn("通过推断纳入候选池", field)


class GenerateInferredExplorationFloorSchemaTest(unittest.TestCase):
    """P15-S3-S3C schema contract: the inferred-only optional ``min_exploration``.

    The plain tool's envelope stays byte-level unchanged (no such property), the
    inferred tool exposes an optional non-negative integer that acts on FINAL
    SELECTION only, and the surfaced description never claims a freshness
    guarantee.
    """

    def test_inferred_schema_exposes_optional_bounded_floor(self) -> None:
        schema = PROVIDER_TOOLS_BY_NAME["generate_inferred_recommendation"].input_schema
        field = schema["properties"]["min_exploration"]
        self.assertEqual(field["type"], "integer")
        self.assertGreaterEqual(field["minimum"], 0)
        self.assertNotIn("min_exploration", schema["required"])
        # Selection-stage semantics must be stated on the wire schema itself.
        self.assertIn("最终选择", field["description"])
        self.assertIn("不改评分/资格/排序", field["description"])

    def test_plain_schema_has_no_min_exploration(self) -> None:
        schema = PROVIDER_TOOLS_BY_NAME["generate_recommendation"].input_schema
        self.assertNotIn("min_exploration", schema["properties"])

    def test_inferred_description_never_promises_freshness(self) -> None:
        desc = _description("generate_inferred_recommendation")
        self.assertIn("min_exploration", desc)
        # S3-S3D rewrite: min_exploration stays freshness-neutral, and the new
        # min_fresh paragraph pins the ONLY freshness truth source.
        self.assertIn("不代表也不会保证「本次刚发现的新歌」", desc)
        self.assertIn("不改变任何评分/资格/排序", desc)
        self.assertIn("普通推荐请求不要传", desc)
        self.assertIn("只以这两个字段为准", desc)

    def test_inferred_description_names_best_effort_shortfall(self) -> None:
        desc = _description("generate_inferred_recommendation")
        self.assertIn("best-effort", desc)
        self.assertIn("尽力而为", desc)


class QueryCatalogDiscoveryStateToolContractTest(unittest.TestCase):
    """P15-S3-S2: catalog-track memory read -- durable facts and derived labels only,
    never a ranking/score/eligibility or a freshness verdict on the live catalog."""

    def test_schema_canonical_id_xor_term_with_bounded_limit(self) -> None:
        schema = dict(PROVIDER_TOOLS_BY_NAME["query_catalog_discovery_state"].input_schema)
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["required"], [])
        self.assertEqual(schema["additionalProperties"], False)
        self.assertEqual(
            set(schema["properties"]), {"canonical_id", "term", "limit"}
        )
        self.assertEqual(schema["properties"]["limit"]["minimum"], 1)
        self.assertEqual(schema["properties"]["limit"]["maximum"], 50)

    def test_description_names_derived_labels_and_memory_boundary(self) -> None:
        desc = _description("query_catalog_discovery_state")
        self.assertIn("never_recommended", desc)
        self.assertIn("previously_recommended", desc)
        # Facts-only memory: no ranking, no score, no eligibility verdicts.
        self.assertIn("不含排名、探索打分或资格判定", desc)
        # Never a freshness conclusion about the catalog; live search stays the
        # freshness authority.
        self.assertIn("不隐含对 Apple Music 目录新鲜度的任何结论", desc)
        self.assertIn("discover_catalog_tracks（实时搜索）", desc)

    def test_schema_has_no_ranking_or_refresh_surface(self) -> None:
        desc = _description("query_catalog_discovery_state")
        schema = PROVIDER_TOOLS_BY_NAME["query_catalog_discovery_state"].input_schema
        for key in schema["properties"]:
            self.assertNotIn("force_refresh", key)
        self.assertNotIn("force_refresh", desc)


class DiscoverLiveSearchContractTest(unittest.TestCase):
    """P15-S3-S2 scope correction: discover_catalog_tracks keeps live-search semantics,
    zero term-reuse or refresh surface -- catalog_track_state is not the catalog boundary."""

    def test_discover_schema_stays_term_and_limit_only(self) -> None:
        properties = PROVIDER_TOOLS_BY_NAME["discover_catalog_tracks"].input_schema["properties"]
        self.assertEqual(set(properties), {"term", "limit"})
        self.assertNotIn("force_refresh", properties)

    def test_discover_description_still_requires_live_search(self) -> None:
        desc = _description("discover_catalog_tracks")
        self.assertIn("实时搜索", PROVIDER_TOOLS_BY_NAME["query_catalog_discovery_state"].description)
        # discover itself must not advertise memory-hit reuse or cached results.
        self.assertNotIn("force_refresh", desc)
        self.assertNotIn("记忆", desc)
        self.assertNotIn("缓存", desc)


class OpenInAppleMusicToolContractTest(unittest.TestCase):
    """P16-S4: the open tool takes only a canonical id; the service resolves Apple's
    real trackViewUrl -- the description must never invite URL composition and must
    name the honest library-only absence."""

    def test_open_takes_only_canonical_id(self) -> None:
        schema = PROVIDER_TOOLS_BY_NAME["open_in_apple_music"].input_schema
        self.assertEqual(set(schema["properties"]), {"canonical_id"})
        self.assertEqual(schema["required"], ["canonical_id"])

    def test_open_description_names_real_resolution_never_composition(self) -> None:
        desc = _description("open_in_apple_music")
        self.assertIn("绝不自行构造", desc)
        self.assertIn("trackViewUrl", desc)
        self.assertIn("url 字段即真实链接", desc)

    def test_open_description_names_library_only_absence(self) -> None:
        desc = _description("open_in_apple_music")
        self.assertIn("apple_music_open_unavailable", desc)
        self.assertIn("没有链接", desc)
        self.assertIn("不得用搜索", desc)


class PreviewBatchStateContractTest(unittest.TestCase):
    """P17-A2: the preview_batch description projects ONE authoritative session
    state; started is derived, never a promise the model may echo."""

    def test_preview_batch_description_makes_session_state_authoritative(self) -> None:
        desc = _description("preview_batch")
        self.assertIn("权威会话状态快照", desc)
        self.assertIn("started 只是 state==running 的派生标志", desc)
        self.assertIn("state=running/completed/failed/cancelled", desc)

    def test_preview_batch_description_rejects_stray_started_claims(self) -> None:
        desc = _description("preview_batch")
        self.assertIn("绝不宣称已启动", desc)
        self.assertIn("started=false 时绝不宣称已启动", desc)
        self.assertIn("如实说明 failure_reason", desc)


if __name__ == "__main__":
    unittest.main()
