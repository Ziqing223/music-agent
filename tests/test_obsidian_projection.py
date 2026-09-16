"""P10.9: Bounded Obsidian projection writer tests (fixture Vaults in temp dirs only).

The real Vault is never touched by this suite; every test builds a synthetic fixture Vault
that replicates the approved anchor structure with placeholder content.
"""

import tempfile
import unittest
from pathlib import Path

from music_agent.obsidian_projection import (
    MUSIC_AGENT_PARTITION,
    ObsidianProjectionConfig,
    ObsidianProjectionError,
    ObsidianProjectionStructureError,
    ObsidianProjectionValidationError,
    ObsidianProjectionWriter,
    ObsidianSurface,
    ObsidianWriteAction,
    FeedbackRecord,
    RecommendationItem,
    RecommendationRecord,
)

HOMEPAGE_TEMPLATE = """# Music Agent

入口。

## 当前状态

- 种子收藏：73 首
- 自动同步：尚未启用
- 数据来源：截图整理

## 导航

- [[02_Profile/音乐偏好画像]]
"""

RECOMMENDATIONS_TEMPLATE = """# 推荐历史

用于防止重复推荐并记录每次推荐的依据。当前尚无推荐记录。

## 记录格式

### YYYY-MM-DD｜推荐主题

| 歌曲 | 艺人 | 匹配理由 | 潜在风险 | 后续反馈 |
|---|---|---|---|---|
| 示例 | 示例艺人 | 强旋律、清透人声 | 速度可能偏慢 | 待反馈 |

记录真实推荐时删除示例行。
"""

FEEDBACK_TEMPLATE = """# 推荐反馈

用于记录对推荐歌曲的明确反馈，并为画像提供修正依据。当前尚无反馈记录。

## 推荐记录方式

### YYYY-MM-DD｜歌曲 — 艺人

- 评分：1–5 / 👍 / 👎
- 喜欢：
- 不喜欢：
- 场景：
- 是否加入收藏：
- 对画像的影响：待累计更多证据后判断

## 更新原则

- 明确表述优先于模型推断。
"""

UNRELATED_PROFILE = """# 音乐偏好画像

> 版本：V0.1

人类维护的推断内容，机器不得改写。
"""

UNRELATED_RULES = """# Music Agent 分区规则

规则文件，机器不得改写。
"""


def recommendation(**kwargs) -> RecommendationRecord:
    defaults = dict(
        run_id="rcm_11111111-1111-4111-8111-111111111111",
        date="2026-08-16",
        theme="夏日旋律",
        items=(
            RecommendationItem("测试歌曲", "测试艺人", "强旋律", "速度偏慢"),
        ),
    )
    defaults.update(kwargs)
    return RecommendationRecord(**defaults)


def feedback(**kwargs) -> FeedbackRecord:
    defaults = dict(
        feedback_id="fbk_22222222-2222-4222-8222-222222222222",
        date="2026-08-16",
        song="测试歌曲",
        artist="测试艺人",
        fields={"评分": "5", "喜欢": "副歌", "场景": "通勤"},
    )
    defaults.update(kwargs)
    return FeedbackRecord(**defaults)


class FixtureVault:
    """Builds a synthetic Vault replicating the approved structure (no user content)."""

    def __init__(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "Vault"
        partition = self.root / MUSIC_AGENT_PARTITION
        (partition / "01_Library").mkdir(parents=True)
        (partition / "02_Profile").mkdir(parents=True)
        (partition / "03_Recommendations").mkdir(parents=True)
        (partition / "04_Feedback").mkdir(parents=True)
        (partition / "05_System").mkdir(parents=True)
        self.homepage = partition / "首页.md"
        self.recommendations = partition / "03_Recommendations" / "推荐历史.md"
        self.feedback = partition / "04_Feedback" / "推荐反馈.md"
        self.homepage.write_text(HOMEPAGE_TEMPLATE, encoding="utf-8")
        self.recommendations.write_text(RECOMMENDATIONS_TEMPLATE, encoding="utf-8")
        self.feedback.write_text(FEEDBACK_TEMPLATE, encoding="utf-8")
        self.unrelated_profile = partition / "02_Profile" / "音乐偏好画像.md"
        self.unrelated_profile.write_text(UNRELATED_PROFILE, encoding="utf-8")
        self.unrelated_rules = partition / "CLAUDE.md"
        self.unrelated_rules.write_text(UNRELATED_RULES, encoding="utf-8")
        self.writer = ObsidianProjectionWriter(ObsidianProjectionConfig(vault_root=self.root))

    def close(self) -> None:
        self.temporary_directory.cleanup()


class ConfigTest(unittest.TestCase):
    def test_rejects_non_path_root(self) -> None:
        with self.assertRaises(ObsidianProjectionValidationError):
            ObsidianProjectionConfig(vault_root="vault")  # type: ignore[arg-type]

    def test_rejects_missing_vault_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = ObsidianProjectionWriter(
                ObsidianProjectionConfig(vault_root=Path(tmp) / "absent")
            )
            with self.assertRaises(ObsidianProjectionStructureError):
                writer.update_homepage_autosync_status("测试")

    def test_rejects_missing_partition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = ObsidianProjectionWriter(ObsidianProjectionConfig(vault_root=Path(tmp)))
            with self.assertRaises(ObsidianProjectionStructureError):
                writer.update_homepage_autosync_status("测试")


class HomepageStatusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vault = FixtureVault()
        self.addCleanup(self.vault.close)

    def test_updates_exactly_one_bullet_and_preserves_everything_else(self) -> None:
        before = self.vault.homepage.read_text(encoding="utf-8")
        result = self.vault.writer.update_homepage_autosync_status("P10 runtime 接入中")
        self.assertEqual(result.action, ObsidianWriteAction.UPDATED)
        self.assertTrue(result.verified)
        self.assertEqual(result.surface, ObsidianSurface.HOMEPAGE_STATUS)
        after = self.vault.homepage.read_text(encoding="utf-8")
        self.assertEqual(
            after,
            before.replace(
                "- 自动同步：尚未启用", "- 自动同步：P10 runtime 接入中", 1
            ),
        )
        # Every unrelated byte remains: profile + rules untouched.
        self.assertEqual(self.vault.unrelated_profile.read_text(encoding="utf-8"), UNRELATED_PROFILE)
        self.assertEqual(self.vault.unrelated_rules.read_text(encoding="utf-8"), UNRELATED_RULES)

    def test_second_identical_update_is_a_noop(self) -> None:
        self.vault.writer.update_homepage_autosync_status("P10 runtime 接入中")
        before = self.vault.homepage.read_text(encoding="utf-8")
        result = self.vault.writer.update_homepage_autosync_status("P10 runtime 接入中")
        self.assertEqual(result.action, ObsidianWriteAction.NOOP_UNCHANGED)
        self.assertEqual(self.vault.homepage.read_text(encoding="utf-8"), before)

    def test_fails_closed_on_missing_bullet(self) -> None:
        content = self.vault.homepage.read_text(encoding="utf-8").replace(
            "- 自动同步：尚未启用", "- 同步：尚未启用", 1
        )
        self.vault.homepage.write_text(content, encoding="utf-8")
        before = self.vault.homepage.read_text(encoding="utf-8")
        with self.assertRaises(ObsidianProjectionStructureError):
            self.vault.writer.update_homepage_autosync_status("新状态")
        self.assertEqual(self.vault.homepage.read_text(encoding="utf-8"), before)

    def test_fails_closed_on_duplicate_bullet(self) -> None:
        content = self.vault.homepage.read_text(encoding="utf-8")
        content = content.replace(
            "- 数据来源：截图整理", "- 数据来源：截图整理\n- 自动同步：另一条", 1
        )
        self.vault.homepage.write_text(content, encoding="utf-8")
        with self.assertRaises(ObsidianProjectionStructureError):
            self.vault.writer.update_homepage_autosync_status("新状态")

    def test_fails_closed_on_missing_section(self) -> None:
        content = self.vault.homepage.read_text(encoding="utf-8").replace("## 当前状态", "## 状态", 1)
        self.vault.homepage.write_text(content, encoding="utf-8")
        with self.assertRaises(ObsidianProjectionStructureError):
            self.vault.writer.update_homepage_autosync_status("新状态")

    def test_rejects_multiline_or_marker_status(self) -> None:
        with self.assertRaises(ObsidianProjectionValidationError):
            self.vault.writer.update_homepage_autosync_status("第一行\n第二行")
        with self.assertRaises(ObsidianProjectionValidationError):
            self.vault.writer.update_homepage_autosync_status("x --> y")


class RecommendationAppendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vault = FixtureVault()
        self.addCleanup(self.vault.close)

    def test_appends_block_before_anchor_and_preserves_rest(self) -> None:
        before = self.vault.recommendations.read_text(encoding="utf-8")
        result = self.vault.writer.append_recommendation_record(recommendation())
        self.assertEqual(result.action, ObsidianWriteAction.APPENDED)
        self.assertTrue(result.verified)
        after = self.vault.recommendations.read_text(encoding="utf-8")
        # The anchor block (format documentation) stays exactly at the end.
        self.assertTrue(after.endswith(before[before.index("## 记录格式"):]))
        self.assertIn("### 2026-08-16｜夏日旋律", after)
        self.assertIn("<!-- rcm_run: rcm_11111111-1111-4111-8111-111111111111 -->", after)
        self.assertIn("| 测试歌曲 | 测试艺人 | 强旋律 | 速度偏慢 |", after)
        self.assertNotIn("当前尚无推荐记录。", after)  # first real record updates the intro
        self.assertIn("已有真实推荐记录，见下方。", after)

    def test_replay_same_run_id_is_noop(self) -> None:
        self.vault.writer.append_recommendation_record(recommendation())
        before = self.vault.recommendations.read_text(encoding="utf-8")
        result = self.vault.writer.append_recommendation_record(recommendation())
        self.assertEqual(result.action, ObsidianWriteAction.NOOP_REPLAY)
        self.assertEqual(self.vault.recommendations.read_text(encoding="utf-8"), before)

    def test_two_different_records_append_without_touching_intro_again(self) -> None:
        self.vault.writer.append_recommendation_record(recommendation())
        second = recommendation(
            run_id="rcm_33333333-3333-4333-8333-333333333333",
            date="2026-08-17",
            theme="夜航",
        )
        self.vault.writer.append_recommendation_record(second)
        after = self.vault.recommendations.read_text(encoding="utf-8")
        self.assertEqual(after.count("### 2026-08-16｜夏日旋律"), 1)
        self.assertEqual(after.count("### 2026-08-17｜夜航"), 1)
        self.assertEqual(after.count("已有真实推荐记录，见下方。"), 1)

    def test_fails_closed_on_missing_or_duplicate_anchor(self) -> None:
        content = self.vault.recommendations.read_text(encoding="utf-8")
        self.vault.recommendations.write_text(content.replace("## 记录格式", "## 格式"), encoding="utf-8")
        before = self.vault.recommendations.read_text(encoding="utf-8")
        with self.assertRaises(ObsidianProjectionStructureError):
            self.vault.writer.append_recommendation_record(recommendation())
        self.assertEqual(self.vault.recommendations.read_text(encoding="utf-8"), before)
        duplicated = content + "\n## 记录格式\n副本\n"
        self.vault.recommendations.write_text(duplicated, encoding="utf-8")
        with self.assertRaises(ObsidianProjectionStructureError):
            self.vault.writer.append_recommendation_record(recommendation())

    def test_record_validation(self) -> None:
        with self.assertRaises(ObsidianProjectionValidationError):
            recommendation(run_id="bad-id")
        with self.assertRaises(ObsidianProjectionValidationError):
            recommendation(date="2026/08/16")
        with self.assertRaises(ObsidianProjectionValidationError):
            recommendation(theme="含\n换行")
        with self.assertRaises(ObsidianProjectionValidationError):
            recommendation(items=())
        with self.assertRaises(ObsidianProjectionValidationError):
            recommendation(items=(RecommendationItem("带|管道", "艺人", "理由", "风险"),))
        with self.assertRaises(ObsidianProjectionValidationError):
            recommendation(items=(RecommendationItem("歌", "艺\n人", "理由", "风险"),))
        with self.assertRaises(ObsidianProjectionValidationError):
            recommendation(items=(RecommendationItem("", "艺人", "理由", "风险"),))


class FeedbackAppendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vault = FixtureVault()
        self.addCleanup(self.vault.close)

    def test_appends_block_before_anchor_and_preserves_rest(self) -> None:
        before = self.vault.feedback.read_text(encoding="utf-8")
        result = self.vault.writer.append_feedback_record(feedback())
        self.assertEqual(result.action, ObsidianWriteAction.APPENDED)
        self.assertTrue(result.verified)
        after = self.vault.feedback.read_text(encoding="utf-8")
        self.assertTrue(after.endswith(before[before.index("## 推荐记录方式"):]))
        self.assertIn("### 2026-08-16｜测试歌曲 — 测试艺人", after)
        self.assertIn("<!-- fbk_obs: fbk_22222222-2222-4222-8222-222222222222 -->", after)
        self.assertIn("- 评分：5", after)
        self.assertIn("- 场景：通勤", after)
        self.assertIn("- 是否加入收藏：", after)  # unanswered fields render empty
        self.assertNotIn("当前尚无反馈记录。", after)
        self.assertIn("已有真实反馈记录，见下方。", after)

    def test_replay_same_feedback_id_is_noop(self) -> None:
        self.vault.writer.append_feedback_record(feedback())
        before = self.vault.feedback.read_text(encoding="utf-8")
        result = self.vault.writer.append_feedback_record(feedback())
        self.assertEqual(result.action, ObsidianWriteAction.NOOP_REPLAY)
        self.assertEqual(self.vault.feedback.read_text(encoding="utf-8"), before)

    def test_unknown_field_refused(self) -> None:
        with self.assertRaises(ObsidianProjectionValidationError):
            feedback(fields={"自定义字段": "x"})
        with self.assertRaises(ObsidianProjectionValidationError):
            feedback(fields={"喜欢": "多\n行"})
        with self.assertRaises(ObsidianProjectionValidationError):
            feedback(feedback_id="not-an-fbk-id")  # wrong namespace prefix

    def test_fails_closed_on_missing_anchor(self) -> None:
        content = self.vault.feedback.read_text(encoding="utf-8").replace(
            "## 推荐记录方式", "## 记录方式", 1
        )
        self.vault.feedback.write_text(content, encoding="utf-8")
        before = self.vault.feedback.read_text(encoding="utf-8")
        with self.assertRaises(ObsidianProjectionStructureError):
            self.vault.writer.append_feedback_record(feedback())
        self.assertEqual(self.vault.feedback.read_text(encoding="utf-8"), before)


class ObsidianCliTest(unittest.TestCase):
    def test_update_status_cli_parses(self) -> None:
        from music_agent.cli import build_parser

        args = build_parser().parse_args(
            ["obsidian", "update-status", "--vault", "/tmp/vault", "--status", "接入中"]
        )
        self.assertEqual(args.obsidian_command, "update-status")
        self.assertEqual(args.vault, Path("/tmp/vault"))
        self.assertEqual(args.status, "接入中")

    def test_update_status_cli_writes_through_production_writer(self) -> None:
        import io
        from contextlib import redirect_stderr
        from unittest.mock import patch

        from music_agent.cli import main

        vault = FixtureVault()
        self.addCleanup(vault.close)
        stdout = io.StringIO()
        with patch("sys.stdout", stdout), redirect_stderr(io.StringIO()):
            exit_code = main([
                "obsidian", "update-status",
                "--vault", str(vault.root),
                "--status", "P10 runtime 接入中；尚未启用常驻同步",
            ])
        self.assertEqual(exit_code, 0)
        self.assertIn("updated homepage_status", stdout.getvalue())
        content = vault.homepage.read_text(encoding="utf-8")
        self.assertIn("- 自动同步：P10 runtime 接入中；尚未启用常驻同步", content)
        self.assertNotIn("尚未启用\n", content)

    def test_update_status_cli_refuses_on_bad_vault(self) -> None:
        import io
        from contextlib import redirect_stderr
        from unittest.mock import patch

        from music_agent.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with patch("sys.stdout", io.StringIO()), redirect_stderr(stderr):
                exit_code = main([
                    "obsidian", "update-status",
                    "--vault", str(Path(tmp) / "absent"),
                    "--status", "接入中",
                ])
            self.assertEqual(exit_code, 1)
            self.assertIn("refused", stderr.getvalue())


class IsolationTest(unittest.TestCase):
    def test_no_surface_touches_user_owned_files(self) -> None:
        vault = FixtureVault()
        self.addCleanup(vault.close)
        vault.writer.update_homepage_autosync_status("P10 runtime 接入中")
        vault.writer.append_recommendation_record(recommendation())
        vault.writer.append_feedback_record(feedback())
        self.assertEqual(vault.unrelated_profile.read_text(encoding="utf-8"), UNRELATED_PROFILE)
        self.assertEqual(vault.unrelated_rules.read_text(encoding="utf-8"), UNRELATED_RULES)
        # No temp files left behind anywhere in the fixture vault.
        leftovers = [p for p in vault.root.rglob("*.tmp")]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
