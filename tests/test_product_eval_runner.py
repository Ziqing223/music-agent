from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from product_eval_runner import (  # noqa: E402
    CorpusIndex,
    CorpusTask,
    build_run_data,
    compute_iteration_delta,
    compute_metrics,
    compute_worktree_state_id,
    derive_overall_result,
    discover_previous_run,
    load_run_schema,
    load_structured_run_history,
    render_scorecard_block,
    render_scorecard_history,
    resolve_previous_run_path,
    update_scorecard,
    validate_run_data,
)


class ProductEvalMetricTest(unittest.TestCase):
    def _case(
        self,
        case_id: str,
        result: str,
        *,
        case_type: str = "Core",
        priority: str = "P0",
        uat: str = "NOT_RUN",
        action_executed: bool = False,
        wrong_action=None,
    ):
        return {
            "eval_case_id": case_id,
            "parent_task_id": None,
            "name": case_id,
            "type": case_type,
            "priority": priority,
            "scope": "task",
            "counts_toward_acceptance_metrics": True,
            "result": result,
            "automated_result": "PASS" if result == "PASS" else ("PARTIAL" if result == "PARTIAL" else ("FAIL" if result == "FAIL" else "NOT_RUN")),
            "owner_uat_result": uat,
            "owner_uat_required_for_case": False,
            "failure_layer": "Execution" if result == "FAIL" else None,
            "root_cause_status": None,
            "wrong_action": wrong_action,
            "action_executed": action_executed,
            "evidence": [],
            "notes": [],
        }

    def test_pass_fail_aggregation(self):
        metrics = compute_metrics([self._case("a", "PASS"), self._case("b", "FAIL")])
        self.assertEqual(metrics["acceptance_pass_rate"]["numerator"], 1)
        self.assertEqual(metrics["acceptance_pass_rate"]["denominator"], 2)
        self.assertEqual(metrics["acceptance_pass_rate"]["rate"], 0.5)

    def test_not_run_is_excluded_from_acceptance_denominator(self):
        metrics = compute_metrics([self._case("a", "PASS"), self._case("b", "NOT_RUN")])
        self.assertEqual(metrics["acceptance_pass_rate"]["denominator"], 1)

    def test_waiting_for_uat_is_excluded_from_acceptance_and_uat_denominators(self):
        waiting = self._case("b", "WAITING_FOR_UAT", uat="WAITING_FOR_UAT")
        waiting["automated_result"] = "PASS"
        metrics = compute_metrics([self._case("a", "PASS"), waiting])
        self.assertEqual(metrics["acceptance_pass_rate"]["denominator"], 1)
        self.assertEqual(metrics["owner_uat_pass_rate"]["denominator"], 0)

    def test_automated_pass_plus_required_owner_uat_fail_is_overall_fail(self):
        self.assertEqual(
            derive_overall_result("PASS", "FAIL", owner_uat_required_for_case=True),
            "FAIL",
        )

    def test_partial_is_in_acceptance_denominator_but_not_pass_numerator(self):
        metrics = compute_metrics([self._case("a", "PASS"), self._case("b", "PARTIAL")])
        self.assertEqual(metrics["acceptance_pass_rate"]["numerator"], 1)
        self.assertEqual(metrics["acceptance_pass_rate"]["denominator"], 2)
        self.assertEqual(metrics["acceptance_pass_rate"]["rate"], 0.5)

    def test_core_filter(self):
        metrics = compute_metrics(
            [self._case("core", "PASS", case_type="Core"), self._case("ext", "FAIL", case_type="Extension")]
        )
        self.assertEqual(metrics["core_acceptance_pass_rate"]["numerator"], 1)
        self.assertEqual(metrics["core_acceptance_pass_rate"]["denominator"], 1)

    def test_p0_filter(self):
        metrics = compute_metrics(
            [self._case("p0", "PASS", priority="P0"), self._case("p1", "FAIL", priority="P1")]
        )
        self.assertEqual(metrics["p0_acceptance_pass_rate"]["numerator"], 1)
        self.assertEqual(metrics["p0_acceptance_pass_rate"]["denominator"], 1)

    def test_wrong_action_rate_uses_only_executed_cases_with_explicit_wrong_action(self):
        metrics = compute_metrics(
            [
                self._case("good", "PASS", action_executed=True, wrong_action=False),
                self._case("wrong", "FAIL", action_executed=True, wrong_action=True),
                self._case("missing", "FAIL", action_executed=False, wrong_action=False),
                self._case("unknown", "PASS", action_executed=True, wrong_action=None),
            ]
        )
        self.assertEqual(metrics["wrong_action_rate"]["numerator"], 1)
        self.assertEqual(metrics["wrong_action_rate"]["denominator"], 2)
        self.assertEqual(metrics["wrong_action_rate"]["rate"], 0.5)

    def test_zero_denominator_metrics_are_na(self):
        metrics = compute_metrics([])
        self.assertIsNone(metrics["acceptance_pass_rate"]["rate"])
        self.assertIsNone(metrics["core_acceptance_pass_rate"]["rate"])
        self.assertIsNone(metrics["p0_acceptance_pass_rate"]["rate"])
        self.assertIsNone(metrics["owner_uat_pass_rate"]["rate"])
        self.assertIsNone(metrics["wrong_action_rate"]["rate"])
        self.assertIsNone(metrics["engineering_regression_pass_rate"]["rate"])


class WorktreeIdentityTest(unittest.TestCase):
    def _corpus(self):
        return CorpusIndex(
            schema_version=1,
            eval_corpus_version="test",
            tasks=(CorpusTask("A", "A", "Core", "P0", False),),
            worktree_scope=("src/**", "eval/**", "README.md"),
            worktree_excluded=("eval/**", "**/__pycache__/**", "*.pyc"),
        )

    def test_worktree_state_identity_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src/a.py").write_text("x = 1\n", encoding="utf-8")
            (root / "README.md").write_text("hello\n", encoding="utf-8")
            first = compute_worktree_state_id(root, self._corpus())
            second = compute_worktree_state_id(root, self._corpus())
            self.assertEqual(first, second)

    def test_eval_files_are_excluded_from_worktree_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "eval").mkdir()
            (root / "src/a.py").write_text("x = 1\n", encoding="utf-8")
            (root / "README.md").write_text("hello\n", encoding="utf-8")
            (root / "eval/result.json").write_text("one\n", encoding="utf-8")
            first = compute_worktree_state_id(root, self._corpus())
            (root / "eval/result.json").write_text("two\n", encoding="utf-8")
            second = compute_worktree_state_id(root, self._corpus())
            self.assertEqual(first, second)


class ProductEvalArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.corpus = cls.root / "eval/product_acceptance_v1.yaml"
        cls.schema = cls.root / "eval/product_eval_run.schema.json"

    def test_run_json_round_trip_and_schema_validation(self):
        run = build_run_data(
            root=self.root,
            corpus_path=self.corpus,
            schema_path=self.schema,
            run_id="TEST-RUN",
            raw_records=[],
            git_head=None,
            git_head_source="test",
            provider="N/A",
            model="N/A",
            database_state="N/A",
            previous_run_path=None,
            engineering_regression={"status": "NOT_RUN", "passed": None, "total": None},
            timestamp="2026-09-16T00:00:00+00:00",
        )
        round_tripped = json.loads(json.dumps(run, ensure_ascii=False))
        validate_run_data(round_tripped, load_run_schema(self.schema))
        self.assertEqual(round_tripped["corpus"]["task_count"], 15)
        self.assertEqual(len([c for c in round_tripped["cases"] if c["scope"] == "task"]), 15)

    def test_scorecard_numbers_are_rendered_from_json_aggregate(self):
        run = build_run_data(
            root=self.root,
            corpus_path=self.corpus,
            schema_path=self.schema,
            run_id="TEST-SCORE",
            raw_records=[
                {
                    "eval_case_id": "ACPT-P22-ROUTE-001",
                    "automated_result": "PASS",
                    "owner_uat_result": "PASS",
                    "owner_uat_required_for_case": True,
                    "action_executed": True,
                    "wrong_action": False,
                },
                {
                    "eval_case_id": "ACPT-P22-CTX-004",
                    "automated_result": "PASS",
                    "owner_uat_result": "FAIL",
                    "owner_uat_required_for_case": True,
                    "failure_layer": "Authority",
                    "action_executed": False,
                    "wrong_action": False,
                },
            ],
            git_head=None,
            git_head_source="test",
            provider="N/A",
            model="N/A",
            database_state="N/A",
            previous_run_path=None,
            engineering_regression={"status": "NOT_RUN", "passed": None, "total": None},
            timestamp="2026-09-16T00:00:00+00:00",
        )
        block = render_scorecard_block(run)
        self.assertIn("Acceptance Pass Rate: **50.00% (1/2)**", block)
        self.assertIn("PASS / FAIL / PARTIAL / NOT RUN / WAITING: **1 / 1 / 0 / 13 / 0**", block)
        with tempfile.TemporaryDirectory() as tmp:
            scorecard = Path(tmp) / "scorecard.md"
            scorecard.write_text("# Score\n\n## 1. Eval Run Identity\n\nHistorical context.\n", encoding="utf-8")
            update_scorecard(scorecard, run)
            updated = scorecard.read_text(encoding="utf-8")
            self.assertIn("50.00% (1/2)", updated)
            self.assertIn("Historical context.", updated)

    def test_historical_markdown_yields_unavailable_delta_without_guessing(self):
        current = compute_metrics([ProductEvalMetricTest()._case("a", "PASS")])
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "old.md"
            old.write_text("PASS = 99 / 100\n", encoding="utf-8")
            delta = compute_iteration_delta(current, old)
            self.assertFalse(delta["available"])
            self.assertIn("not structured JSON", delta["reason"])


class ProductEvalAutomationTest(unittest.TestCase):
    @staticmethod
    def _minimal_run(run_id: str, *, passed: int, total: int = 15, uat_passed: int | None = None, engineering=(None, None, "NOT_RUN"), delta=None):
        uat_passed = passed if uat_passed is None else uat_passed
        eng_passed, eng_total, eng_status = engineering
        return {
            "run_metadata": {"run_id": run_id},
            "corpus": {"task_count": total},
            "metrics": {
                "acceptance_pass_rate": {"numerator": passed, "denominator": passed, "rate": 1.0 if passed else None},
                "owner_uat_pass_rate": {"numerator": uat_passed, "denominator": uat_passed, "rate": 1.0 if uat_passed else None},
                "engineering_regression_pass_rate": {
                    "numerator": eng_passed or 0,
                    "denominator": eng_total or 0,
                    "rate": None if not eng_total else eng_passed / eng_total,
                    "status": eng_status,
                },
            },
            "iteration_delta": delta or {"available": False, "reason": "fixture"},
        }

    def test_auto_previous_finds_eval_05_for_eval_06(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            (runs / "P22-EVAL-04.json").write_text("{}", encoding="utf-8")
            expected = runs / "P22-EVAL-05.json"
            expected.write_text("{}", encoding="utf-8")
            self.assertEqual(discover_previous_run(runs, "P22-EVAL-06"), expected)

    def test_auto_previous_excludes_current_and_future_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            previous = runs / "P22-EVAL-05.json"
            previous.write_text("{}", encoding="utf-8")
            (runs / "P22-EVAL-06.json").write_text("{}", encoding="utf-8")
            (runs / "P22-EVAL-07.json").write_text("{}", encoding="utf-8")
            self.assertEqual(discover_previous_run(runs, "P22-EVAL-06"), previous)

    def test_auto_previous_no_structured_run_is_legal(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            (runs / "P22-EVAL-05.md").write_text("historical markdown", encoding="utf-8")
            self.assertIsNone(discover_previous_run(runs, "P22-EVAL-06"))

    def test_explicit_previous_path_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            runs.mkdir()
            explicit = Path(tmp) / "custom.json"
            explicit.write_text("{}", encoding="utf-8")
            resolved = resolve_previous_run_path(str(explicit), runs_dir=runs, current_run_id="P22-EVAL-06")
            self.assertEqual(resolved, explicit.resolve())

    def test_scorecard_execution_coverage_is_explicit(self):
        root = Path(__file__).resolve().parents[1]
        run = build_run_data(
            root=root,
            corpus_path=root / "eval/product_acceptance_v1.yaml",
            schema_path=root / "eval/product_eval_run.schema.json",
            run_id="P22-EVAL-99",
            raw_records=[
                {
                    "eval_case_id": "ACPT-P22-ROUTE-001",
                    "automated_result": "PASS",
                    "owner_uat_result": "PASS",
                    "owner_uat_required_for_case": True,
                },
                {
                    "eval_case_id": "ACPT-P22-CTX-004",
                    "automated_result": "PASS",
                    "owner_uat_result": "PASS",
                    "owner_uat_required_for_case": True,
                },
            ],
            git_head=None,
            git_head_source="test",
            provider="N/A",
            model="N/A",
            database_state="N/A",
            previous_run_path=None,
            engineering_regression={"status": "NOT_RUN", "passed": None, "total": None},
            timestamp="2026-09-16T00:00:00+00:00",
        )
        block = render_scorecard_block(run)
        self.assertIn("Execution Coverage: **13.33% (2/15)**", block)
        self.assertIn("Acceptance Pass Rate: **100.00% (2/2)**", block)
        self.assertIn("not whole-corpus coverage", block)

    def test_iteration_history_is_generated_from_structured_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            run4 = self._minimal_run("P22-EVAL-04", passed=1)
            run5 = self._minimal_run(
                "P22-EVAL-05",
                passed=2,
                engineering=(5421, 5421, "PASS"),
                delta={
                    "available": True,
                    "fixed_failures_count": 1,
                    "new_failures_count": 0,
                },
            )
            (runs_dir / "P22-EVAL-04.json").write_text(json.dumps(run4), encoding="utf-8")
            (runs_dir / "P22-EVAL-05.json").write_text(json.dumps(run5), encoding="utf-8")
            history = load_structured_run_history(runs_dir)
            rendered = render_scorecard_history(history)
            self.assertLess(rendered.index("P22-EVAL-04"), rendered.index("P22-EVAL-05"))
            self.assertIn("13.33% (2/15)", rendered)
            self.assertIn("5421/5421", rendered)
            self.assertIn("| 1 | 0 |", rendered)

    def test_historical_eval_04_and_05_artifacts_match_frozen_hashes(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "eval/evidence/P22-EVAL-04-results.json": "f000d8df7ed5ed14386000eef33e9bc1185c5519d4ba6e814c4d6fbb42621ed1",
            "eval/evidence/P22-EVAL-05-results.json": "5e34de55eb7139ee1635fdebeda258fc4702eb449e03c9bdfc177647ddba4fd7",
            "eval/runs/P22-EVAL-04.json": "7c4fb91af1164482ce46666856ec96e607218da241a6d0f97c92fd3f7ad49bda",
            "eval/runs/P22-EVAL-04.md": "31adb222b01a09fb3f8be95ec8d2e0b2d630821fbfc12c948975fb1981c215fc",
            "eval/runs/P22-EVAL-05.json": "bcc9caa863333f24ea54a83ea34992cf8e3a121add46b82e3ac3795ada583341",
            "eval/runs/P22-EVAL-05.md": "fd73239037532b54a0c4fb4b0695931f18b4d332fa1229e8a6de4ad6154ba573",
        }
        for rel, digest in expected.items():
            actual = hashlib.sha256((root / rel).read_bytes()).hexdigest()
            self.assertEqual(actual, digest, rel)


if __name__ == "__main__":
    unittest.main()
