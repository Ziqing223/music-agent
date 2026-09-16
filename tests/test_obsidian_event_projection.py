"""P10.10: Durable-event -> Vault projection and favorites-ingestion tests.

All Vault reads/writes use synthetic fixture Vaults in temp dirs; the real Vault is never
touched. Recommendation runs and feedback observations are REAL production-pipeline
results persisted to a temp store, so the projections exercise genuine durable events.
"""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from music_agent.candidate_generation import generate_candidates
from music_agent.feedback_contract import (
    FeedbackKind,
    FeedbackRecommendationReference,
    FeedbackSourceReference,
    assemble_feedback_observation,
    generate_feedback_id,
)
from music_agent.feedback_history_repository import FeedbackHistoryRepository
from music_agent.obsidian_event_projection import (
    ObsidianEventProjectionError,
    build_feedback_record,
    build_recommendation_record,
)
from music_agent.preference_attribution import (
    PreferenceState,
    PreferenceTargetKind,
    PreferenceTargetReference,
)
from music_agent.preference_propagation import DerivedPreference
from music_agent.preference_strength import PreferenceStrength
from music_agent.recommendation_contract import (
    PreferenceInput,
    RecommendationContext,
    RecommendationItem,
    RecommendationRequest,
    RecommendedItemKind,
    assemble_recommendation_result,
    generate_run_id,
)
from music_agent.recommendation_history_repository import RecommendationHistoryRepository
from music_agent.recommendation_ranking import rank_items
from music_agent.recommendation_scoring import score_candidate
from music_agent.repository import CanonicalRepository
from tools.ingest_favorites import CSV_RELATIVE_PATH, ingest as ingest_favorites

from music_agent.obsidian_projection import MUSIC_AGENT_PARTITION

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_music_model.json"
TRACK_A = "trk_11111111-1111-4111-8111-111111111111"
NOW = datetime(2026, 8, 16, 0, 0, 0, tzinfo=timezone.utc)
PRODUCED_AT = datetime(2026, 8, 16, 12, 30, 0, tzinfo=timezone.utc)
OBSERVED_AT = datetime(2026, 8, 16, 14, 0, 0, tzinfo=timezone.utc)


def _target(kind: PreferenceTargetKind, target_id: str) -> PreferenceTargetReference:
    return PreferenceTargetReference(kind, target_id)


def _context() -> RecommendationContext:
    return RecommendationContext(
        NOW,
        (
            PreferenceInput.from_direct(
                DerivedPreference(
                    _target(PreferenceTargetKind.TRACK, TRACK_A),
                    PreferenceStrength(PreferenceState.POSITIVE, 0.9),
                )
            ),
        ),
    )


def _save_real_run(database_path: Path) -> str:
    """Run the PRODUCTION recommendation pipeline and persist the real result."""
    context = _context()
    request = RecommendationRequest(context, RecommendedItemKind.TRACK, limit=5)
    candidates = [
        c for c in generate_candidates(context, request.recommended_kind)
        if c.eligibility.value == "eligible"
    ]
    items = [
        RecommendationItem(candidate, score_candidate(candidate, context))
        for candidate in candidates
    ]
    result = assemble_recommendation_result(
        request,
        rank_items(items),
        run_id=generate_run_id(),
        produced_at=PRODUCED_AT,
    )
    with RecommendationHistoryRepository(database_path) as history:
        history.save_result(result)
    return result.run_id


class IngestFavoritesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.vault_root = Path(self.temporary_directory.name) / "Vault"
        library = self.vault_root / CSV_RELATIVE_PATH.parent
        library.mkdir(parents=True)
        (library / "favorites.csv").write_text(
            "title,artist,language,category\n"
            "测试歌曲一,艺人甲,中文,城市华语抒情\n"
            "测试歌曲二,艺人乙,日语,日系青春叙事\n"
            "测试歌曲一,艺人甲,中文,城市华语抒情\n",  # duplicate row
            encoding="utf-8",
        )
        self.database_path = Path(self.temporary_directory.name) / "store.db"

    def test_ingest_creates_canonical_tracks_and_artists_idempotently(self) -> None:
        first = ingest_favorites(self.vault_root, self.database_path)
        # The duplicate CSV row is skipped within the same run.
        self.assertEqual(first, {"added": 2, "skipped": 1, "artists_added": 2, "total": 3})
        with CanonicalRepository(self.database_path) as repository:
            model = repository.load_model()
            self.assertEqual(len(model["tracks"]), 2)
            self.assertEqual(len(model["artists"]), 2)
            track = next(t for t in model["tracks"] if t["name"] == "测试歌曲一")
            self.assertEqual(track["external_ids"]["apple_music_persistent_id"], None)
            self.assertEqual(track["genres"], ["城市华语抒情"])
            artist_id = track["artist_ids"][0]
            artist = next(a for a in model["artists"] if a["id"] == artist_id)
            self.assertEqual(artist["name"], "艺人甲")
        second = ingest_favorites(self.vault_root, self.database_path)
        self.assertEqual(second, {"added": 0, "skipped": 3, "artists_added": 0, "total": 3})
        with CanonicalRepository(self.database_path) as repository:
            self.assertEqual(len(repository.load_model()["tracks"]), 2)

    def test_missing_csv_fails_clearly(self) -> None:
        with self.assertRaises(FileNotFoundError):
            ingest_favorites(Path(self.temporary_directory.name) / "absent", self.database_path)


class RecommendationProjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))
        self.run_id = _save_real_run(self.database_path)

    def test_record_built_from_real_run(self) -> None:
        record = build_recommendation_record(self.database_path, self.run_id, theme="周末推荐")
        self.assertEqual(record.run_id, self.run_id)
        self.assertEqual(record.date, "2026-08-16")
        self.assertEqual(record.theme, "周末推荐")
        self.assertTrue(record.items)
        for item in record.items:
            self.assertTrue(item.song)
            self.assertTrue(item.artist)
            self.assertRegex(item.match_reason, r"^匹配度 \d+%$")
            self.assertEqual(item.risk, "待反馈")
        # The fixture's artists resolve to their canonical names (joined by 、).
        known = {"Artist Alpha", "Artist Beta", "Artist Gamma"}
        for item in record.items:
            self.assertTrue(set(item.artist.split("、")) <= known)

    def test_unknown_run_fails_closed(self) -> None:
        with self.assertRaises(ObsidianEventProjectionError):
            build_recommendation_record(
                self.database_path, "rcm_00000000-0000-4000-8000-000000000000"
            )


class FeedbackProjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
        self.run_id = _save_real_run(self.database_path)

    def _save_observation(
        self, kind: FeedbackKind, *, target: bool = True
    ) -> str:
        recommendation = None
        if not target:
            with RecommendationHistoryRepository(self.database_path) as history:
                run = history.get_result(self.run_id)
            first_candidate = run.items[0].candidate.candidate_id
            recommendation = FeedbackRecommendationReference(self.run_id, first_candidate)
        observation = assemble_feedback_observation(
            feedback_id=generate_feedback_id(),
            kind=kind,
            source=FeedbackSourceReference("apple_music", "live-chat"),
            observed_at=OBSERVED_AT,
            target=_target(PreferenceTargetKind.TRACK, TRACK_A) if target else None,
            recommendation=recommendation,
        )
        with FeedbackHistoryRepository(self.database_path) as history:
            history.save_observation(observation)
        return observation.feedback_id

    def test_liked_observation_maps_fields_and_identity(self) -> None:
        feedback_id = self._save_observation(FeedbackKind.LIKED)
        record = build_feedback_record(self.database_path, feedback_id)
        self.assertEqual(record.feedback_id, feedback_id)
        self.assertEqual(record.date, "2026-08-16")
        self.assertEqual(record.song, "Synthetic Duet")
        self.assertEqual(record.fields["喜欢"], "是")

    def test_skipped_is_not_dislike(self) -> None:
        feedback_id = self._save_observation(FeedbackKind.SKIPPED)
        record = build_feedback_record(self.database_path, feedback_id)
        self.assertEqual(record.fields["场景"], "跳过（跳过≠不喜欢）")
        self.assertNotIn("不喜欢", record.fields)

    def test_recommendation_referenced_observation_resolves_subject(self) -> None:
        feedback_id = self._save_observation(FeedbackKind.COMPLETED, target=False)
        record = build_feedback_record(self.database_path, feedback_id)
        self.assertEqual(record.fields["喜欢"], "完整听完")
        self.assertNotEqual(record.song, "未知歌曲")  # resolved from the real run's candidate

    def test_unknown_feedback_fails_closed(self) -> None:
        with self.assertRaises(ObsidianEventProjectionError):
            build_feedback_record(
                self.database_path, "fbk_00000000-0000-4000-8000-000000000000"
            )


class ObsidianAppendCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.vault_root = Path(self.temporary_directory.name) / "Vault"
        partition = self.vault_root / MUSIC_AGENT_PARTITION
        (partition / "03_Recommendations").mkdir(parents=True)
        (partition / "04_Feedback").mkdir(parents=True)
        (partition / "首页.md").write_text(
            "# Music Agent\n\n## 当前状态\n\n- 自动同步：尚未启用\n",
            encoding="utf-8",
        )
        (partition / "03_Recommendations" / "推荐历史.md").write_text(
            "# 推荐历史\n\n用于防止重复推荐并记录每次推荐的依据。当前尚无推荐记录。\n\n"
            "## 记录格式\n\n### 示例\n\n| 歌曲 | 艺人 | 匹配理由 | 潜在风险 | 后续反馈 |\n|---|---|---|---|---|\n"
            "| 示例 | 示例艺人 | 强旋律、清透人声 | 速度可能偏慢 | 待反馈 |\n",
            encoding="utf-8",
        )
        (partition / "04_Feedback" / "推荐反馈.md").write_text(
            "# 推荐反馈\n\n用于记录对推荐歌曲的明确反馈。当前尚无反馈记录。\n\n"
            "## 推荐记录方式\n\n### 示例\n\n- 评分：1–5 / 👍 / 👎\n",
            encoding="utf-8",
        )
        self.database_path = Path(self.temporary_directory.name) / "store.db"
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        with CanonicalRepository(self.database_path) as repository:
            repository.save_model(fixture)
        self.run_id = _save_real_run(self.database_path)

    def test_append_recommendation_cli_writes_through_production_writer(self) -> None:
        import io
        from contextlib import redirect_stderr
        from unittest.mock import patch

        from music_agent.cli import main

        stdout = io.StringIO()
        with patch("sys.stdout", stdout), redirect_stderr(io.StringIO()):
            exit_code = main([
                "obsidian", "append-recommendation",
                "--vault", str(self.vault_root),
                "--db", str(self.database_path),
                "--run-id", self.run_id,
            ])
        self.assertEqual(exit_code, 0)
        self.assertIn("appended recommendation_history", stdout.getvalue())
        content = (self.vault_root / MUSIC_AGENT_PARTITION / "03_Recommendations" / "推荐历史.md").read_text(encoding="utf-8")
        self.assertIn(f"<!-- rcm_run: {self.run_id} -->", content)
        self.assertIn("| 歌曲 | 艺人 | 匹配理由 | 潜在风险 | 后续反馈 |", content)
        # Re-running the same run id is replay-safe (no-op).
        with patch("sys.stdout", io.StringIO()), redirect_stderr(io.StringIO()):
            exit_code = main([
                "obsidian", "append-recommendation",
                "--vault", str(self.vault_root),
                "--db", str(self.database_path),
                "--run-id", self.run_id,
            ])
        self.assertEqual(exit_code, 0)
        self.assertEqual(content.count(f"<!-- rcm_run: {self.run_id} -->"), 1)

    def test_append_recommendation_refuses_unknown_run(self) -> None:
        import io
        from contextlib import redirect_stderr
        from unittest.mock import patch

        from music_agent.cli import main

        stderr = io.StringIO()
        with patch("sys.stdout", io.StringIO()), redirect_stderr(stderr):
            exit_code = main([
                "obsidian", "append-recommendation",
                "--vault", str(self.vault_root),
                "--db", str(self.database_path),
                "--run-id", "rcm_00000000-0000-4000-8000-000000000000",
            ])
        self.assertEqual(exit_code, 1)
        self.assertIn("refused", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
