#!/usr/bin/env python3
"""Quantitative Product Eval recording and aggregation helpers.

This module is Eval infrastructure only. It reads the fixed Product Acceptance corpus,
records explicit case evidence, computes strict quantitative metrics, and renders JSON /
Markdown outputs. It deliberately does not execute Music Agent product behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import platform as platform_module
import re
import subprocess
import sys
from typing import Any, Iterable

from jsonschema import Draft202012Validator


RESULT_VALUES = {"PASS", "FAIL", "PARTIAL", "NOT_RUN", "WAITING_FOR_UAT"}
AUTOMATED_RESULT_VALUES = {"PASS", "FAIL", "PARTIAL", "NOT_RUN"}
OWNER_UAT_RESULT_VALUES = {"PASS", "FAIL", "PARTIAL", "NOT_RUN", "WAITING_FOR_UAT"}
EVALUABLE_RESULTS = {"PASS", "FAIL", "PARTIAL"}
FAILURE_LAYERS = {"Intent", "Context", "Authority", "Selection", "Tool", "Execution", "Readback", "Presentation", "Observability", "Eval Infrastructure"}
SCORECARD_START = "<!-- PRODUCT_EVAL_GENERATED_START -->"
SCORECARD_END = "<!-- PRODUCT_EVAL_GENERATED_END -->"
SCORECARD_HISTORY_START = "<!-- PRODUCT_EVAL_HISTORY_START -->"
SCORECARD_HISTORY_END = "<!-- PRODUCT_EVAL_HISTORY_END -->"
EVAL_RUN_RE = re.compile(r"^P22-EVAL-(\d+)$")


def eval_run_number(run_id: str) -> int | None:
    match = EVAL_RUN_RE.fullmatch(run_id)
    return int(match.group(1)) if match else None


def discover_previous_run(runs_dir: Path, current_run_id: str) -> Path | None:
    """Return the nearest earlier structured P22 Eval JSON run.

    Only ``P22-EVAL-<N>.json`` files participate. The current run is always
    excluded, future run numbers are never selected, and Markdown is ignored.
    """

    current_number = eval_run_number(current_run_id)
    if current_number is None:
        return None
    candidates: list[tuple[int, Path]] = []
    if not runs_dir.exists():
        return None
    for path in runs_dir.glob("P22-EVAL-*.json"):
        number = eval_run_number(path.stem)
        if number is None or number >= current_number:
            continue
        candidates.append((number, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def resolve_previous_run_path(
    previous_run: str | Path | None,
    *,
    runs_dir: Path,
    current_run_id: str,
) -> Path | None:
    """Resolve an explicit previous run or auto-discover the nearest earlier run."""

    if previous_run is None or str(previous_run).strip().lower() == "auto":
        return discover_previous_run(runs_dir, current_run_id)
    return Path(previous_run).expanduser().resolve()


def load_structured_run_history(runs_dir: Path, *, current_run: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Load structured P22 Eval JSON runs in numeric order.

    A matching JSON file that cannot be parsed is an Eval-infrastructure error; it
    is not silently skipped. ``current_run`` replaces a same-id on-disk record so
    scorecard generation can be deterministic during a run write.
    """

    by_number: dict[int, dict[str, Any]] = {}
    if runs_dir.exists():
        for path in runs_dir.glob("P22-EVAL-*.json"):
            number = eval_run_number(path.stem)
            if number is None:
                continue
            try:
                run = json.loads(path.read_text(encoding="utf-8"))
                if run["run_metadata"]["run_id"] != path.stem:
                    raise ValueError("run_id does not match filename")
                run["metrics"]["acceptance_pass_rate"]
                run["corpus"]["task_count"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid structured Eval history file {path}: {exc}") from exc
            by_number[number] = run
    if current_run is not None:
        run_id = str(current_run.get("run_metadata", {}).get("run_id", ""))
        number = eval_run_number(run_id)
        if number is not None:
            by_number[number] = current_run
    return [by_number[number] for number in sorted(by_number)]


@dataclass(frozen=True)
class CorpusTask:
    task_id: str
    name: str
    type: str
    priority: str
    owner_uat_required: bool | str


@dataclass(frozen=True)
class CorpusIndex:
    schema_version: int
    eval_corpus_version: str
    tasks: tuple[CorpusTask, ...]
    worktree_scope: tuple[str, ...]
    worktree_excluded: tuple[str, ...]

    @property
    def tasks_by_id(self) -> dict[str, CorpusTask]:
        return {task.task_id: task for task in self.tasks}


def _scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        try:
            return json.loads(value) if value.startswith('"') else value[1:-1].replace("''", "'")
        except Exception:
            return value[1:-1]
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def load_corpus_index(path: Path) -> CorpusIndex:
    """Read the small corpus index needed by the runner without adding a YAML dependency.

    The full YAML remains the Product Acceptance contract. This loader intentionally reads
    only stable indexing fields needed for aggregation: corpus version, task identity/type/
    priority/UAT policy, and the existing worktree identity scope/exclusions.
    """

    lines = path.read_text(encoding="utf-8").splitlines()
    schema_version: int | None = None
    corpus_version: str | None = None
    tasks: list[CorpusTask] = []
    current: dict[str, Any] | None = None
    scope: list[str] = []
    excluded: list[str] = []
    in_worktree = False
    list_target: list[str] | None = None

    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw.startswith("schema_version:"):
            schema_version = int(_scalar(raw.split(":", 1)[1]))
            continue
        if raw.startswith("eval_corpus_version:"):
            corpus_version = str(_scalar(raw.split(":", 1)[1]))
            continue

        if raw.startswith("  worktree_state_id:"):
            in_worktree = True
            list_target = None
            continue
        if in_worktree:
            if raw.startswith("    scope:"):
                list_target = scope
                continue
            if raw.startswith("    excluded:"):
                list_target = excluded
                continue
            if raw.startswith("    construction:"):
                in_worktree = False
                list_target = None
                continue
            if list_target is not None and raw.startswith("    - "):
                list_target.append(str(_scalar(raw.split("- ", 1)[1])))
                continue

        if raw.startswith("- task_id:"):
            if current is not None:
                tasks.append(_task_from_mapping(current))
            current = {"task_id": _scalar(raw.split(":", 1)[1])}
            continue
        if current is not None and raw.startswith("  ") and not raw.startswith("    "):
            key, sep, value = stripped.partition(":")
            if sep and key in {"name", "type", "priority", "owner_uat_required"}:
                current[key] = _scalar(value)

    if current is not None:
        tasks.append(_task_from_mapping(current))

    if schema_version is None or corpus_version is None:
        raise ValueError("corpus is missing schema_version or eval_corpus_version")
    if not tasks:
        raise ValueError("corpus contains no tasks")
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("corpus contains duplicate task_id values")
    if not scope:
        raise ValueError("corpus is missing worktree_state_id.scope")
    return CorpusIndex(
        schema_version=schema_version,
        eval_corpus_version=corpus_version,
        tasks=tuple(tasks),
        worktree_scope=tuple(scope),
        worktree_excluded=tuple(excluded),
    )


def _task_from_mapping(mapping: dict[str, Any]) -> CorpusTask:
    missing = [key for key in ("task_id", "name", "type", "priority", "owner_uat_required") if key not in mapping]
    if missing:
        raise ValueError(f"corpus task {mapping.get('task_id', '<unknown>')} missing {', '.join(missing)}")
    return CorpusTask(
        task_id=str(mapping["task_id"]),
        name=str(mapping["name"]),
        type=str(mapping["type"]),
        priority=str(mapping["priority"]),
        owner_uat_required=mapping["owner_uat_required"],
    )


def _matches_exclusion(rel: str, excluded_patterns: Iterable[str]) -> bool:
    parts = rel.split("/")
    if any(part in {".git", "__pycache__", ".build", ".pytest_cache"} for part in parts):
        return True
    if rel.startswith("eval/") or rel == "eval":
        return True
    if rel.endswith(".pyc") or Path(rel).name == ".DS_Store":
        return True
    for pattern in excluded_patterns:
        if pattern.startswith("reports/"):
            continue
        if fnmatch.fnmatch(rel, pattern):
            return True
    return False


def _scope_files(root: Path, scope_patterns: Iterable[str], excluded_patterns: Iterable[str]) -> list[Path]:
    files: dict[str, Path] = {}
    for pattern in scope_patterns:
        if pattern.endswith("/**"):
            base = root / pattern[:-3]
            if not base.exists():
                continue
            candidates = base.rglob("*")
        else:
            candidates = (root / pattern,)
        for candidate in candidates:
            if not candidate.is_file():
                continue
            rel = candidate.relative_to(root).as_posix()
            if _matches_exclusion(rel, excluded_patterns):
                continue
            files[rel] = candidate
    return [files[key] for key in sorted(files)]


def compute_worktree_state_id(root: Path, corpus: CorpusIndex) -> str:
    manifest = bytearray()
    for path in _scope_files(root, corpus.worktree_scope, corpus.worktree_excluded):
        rel = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.extend(rel.encode("utf-8"))
        manifest.extend(b"\0")
        manifest.extend(digest.encode("ascii"))
        manifest.extend(b"\n")
    return "sha256-manifest-v1:" + hashlib.sha256(bytes(manifest)).hexdigest()


def read_project_version(pyproject_path: Path) -> str:
    text = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', text, re.MULTILINE)
    if not match:
        raise ValueError("project version not found in pyproject.toml")
    return match.group(1)


def detect_git_head(root: Path) -> tuple[str | None, str]:
    if not (root / ".git").exists():
        return None, "unavailable: .git metadata absent"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, f"unavailable: {exc}"
    return result.stdout.strip(), "detected from repository"


def derive_overall_result(
    automated_result: str,
    owner_uat_result: str,
    *,
    owner_uat_required_for_case: bool,
) -> str:
    if automated_result not in AUTOMATED_RESULT_VALUES:
        raise ValueError(f"invalid automated_result: {automated_result}")
    if owner_uat_result not in OWNER_UAT_RESULT_VALUES:
        raise ValueError(f"invalid owner_uat_result: {owner_uat_result}")
    if automated_result == "FAIL" or owner_uat_result == "FAIL":
        return "FAIL"
    if automated_result == "NOT_RUN":
        return "PASS" if owner_uat_result == "PASS" else "NOT_RUN"
    if automated_result == "PARTIAL" or owner_uat_result == "PARTIAL":
        return "PARTIAL"
    if automated_result == "PASS":
        if owner_uat_required_for_case and owner_uat_result != "PASS":
            return "WAITING_FOR_UAT"
        return "PASS"
    raise AssertionError("unreachable result combination")


def _normalize_case_record(
    raw: dict[str, Any],
    *,
    corpus: CorpusIndex,
) -> dict[str, Any]:
    tasks = corpus.tasks_by_id
    case_id = str(raw.get("eval_case_id", "")).strip()
    parent_id = raw.get("parent_task_id")
    scope = str(raw.get("scope", "task"))
    if scope not in {"task", "scenario"}:
        raise ValueError(f"{case_id}: scope must be task or scenario")
    if scope == "task":
        if case_id not in tasks:
            raise ValueError(f"unknown top-level eval_case_id: {case_id}")
        parent_id = None
        task = tasks[case_id]
        name = task.name
        case_type = task.type
        priority = task.priority
    else:
        if not parent_id or str(parent_id) not in tasks:
            raise ValueError(f"{case_id}: scenario parent_task_id must reference a corpus task")
        task = tasks[str(parent_id)]
        name = str(raw.get("name") or case_id)
        case_type = str(raw.get("type") or task.type)
        priority = str(raw.get("priority") or task.priority)

    automated = str(raw.get("automated_result", "NOT_RUN"))
    owner_uat = str(raw.get("owner_uat_result", "NOT_RUN"))
    uat_required = bool(raw.get("owner_uat_required_for_case", False))
    result = derive_overall_result(
        automated,
        owner_uat,
        owner_uat_required_for_case=uat_required,
    )
    failure_layer = raw.get("failure_layer")
    if result == "FAIL" and not failure_layer:
        raise ValueError(f"{case_id}: FAIL result requires failure_layer")
    if failure_layer is not None and failure_layer not in FAILURE_LAYERS:
        raise ValueError(f"{case_id}: unknown failure_layer {failure_layer!r}")
    wrong_action = raw.get("wrong_action")
    if wrong_action not in {None, True, False}:
        raise ValueError(f"{case_id}: wrong_action must be true, false, or null")
    action_executed = bool(raw.get("action_executed", False))
    if wrong_action is True and not action_executed:
        raise ValueError(f"{case_id}: wrong_action=true requires action_executed=true")

    evidence = raw.get("evidence", [])
    if not isinstance(evidence, list):
        raise ValueError(f"{case_id}: evidence must be a list")
    notes = raw.get("notes", [])
    if isinstance(notes, str):
        notes = [notes]
    if not isinstance(notes, list):
        raise ValueError(f"{case_id}: notes must be a list or string")

    return {
        "eval_case_id": case_id,
        "parent_task_id": str(parent_id) if parent_id else None,
        "name": name,
        "type": case_type,
        "priority": priority,
        "scope": scope,
        "counts_toward_acceptance_metrics": scope == "task",
        "result": result,
        "automated_result": automated,
        "owner_uat_result": owner_uat,
        "owner_uat_required_for_case": uat_required,
        "failure_layer": failure_layer,
        "root_cause_status": raw.get("root_cause_status"),
        "wrong_action": wrong_action,
        "action_executed": action_executed,
        "evidence": evidence,
        "notes": [str(item) for item in notes],
    }


def build_case_records(raw_records: list[dict[str, Any]], corpus: CorpusIndex) -> list[dict[str, Any]]:
    normalized = [_normalize_case_record(item, corpus=corpus) for item in raw_records]
    seen = set()
    for item in normalized:
        key = item["eval_case_id"]
        if key in seen:
            raise ValueError(f"duplicate eval_case_id in run input: {key}")
        seen.add(key)

    top_level_seen = {item["eval_case_id"] for item in normalized if item["scope"] == "task"}
    for task in corpus.tasks:
        if task.task_id in top_level_seen:
            continue
        normalized.append(
            _normalize_case_record(
                {
                    "eval_case_id": task.task_id,
                    "scope": "task",
                    "automated_result": "NOT_RUN",
                    "owner_uat_result": "NOT_RUN",
                    "owner_uat_required_for_case": False,
                    "notes": ["Not executed in this run; absence is not treated as failure."],
                },
                corpus=corpus,
            )
        )

    task_order = {task.task_id: index for index, task in enumerate(corpus.tasks)}
    return sorted(
        normalized,
        key=lambda item: (
            task_order.get(item["parent_task_id"] or item["eval_case_id"], 10_000),
            0 if item["scope"] == "task" else 1,
            item["eval_case_id"],
        ),
    )


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": None if denominator == 0 else numerator / denominator,
    }


def _acceptance_rate(cases: list[dict[str, Any]], predicate=lambda _: True) -> dict[str, Any]:
    selected = [
        case
        for case in cases
        if case["counts_toward_acceptance_metrics"]
        and predicate(case)
        and case["result"] in EVALUABLE_RESULTS
    ]
    return _rate(sum(case["result"] == "PASS" for case in selected), len(selected))


def _result_counts(cases: list[dict[str, Any]], *, top_level_only: bool) -> dict[str, int]:
    result = {value: 0 for value in ("PASS", "FAIL", "PARTIAL", "NOT_RUN", "WAITING_FOR_UAT")}
    for case in cases:
        if top_level_only and not case["counts_toward_acceptance_metrics"]:
            continue
        if not top_level_only and case["scope"] != "scenario":
            continue
        result[case["result"]] += 1
    return result


def compute_metrics(
    cases: list[dict[str, Any]],
    *,
    engineering_regression: dict[str, Any] | None = None,
) -> dict[str, Any]:
    acceptance = _acceptance_rate(cases)
    core = _acceptance_rate(cases, lambda case: case["type"] == "Core")
    p0 = _acceptance_rate(cases, lambda case: case["priority"] == "P0")

    uat_cases = [
        case
        for case in cases
        if case["counts_toward_acceptance_metrics"] and case["owner_uat_result"] in {"PASS", "FAIL", "PARTIAL"}
    ]
    owner_uat = _rate(sum(case["owner_uat_result"] == "PASS" for case in uat_cases), len(uat_cases))

    wrong_action_cases = [
        case for case in cases if case["action_executed"] and isinstance(case["wrong_action"], bool)
    ]
    wrong_action = _rate(sum(case["wrong_action"] is True for case in wrong_action_cases), len(wrong_action_cases))

    engineering = engineering_regression or {"status": "NOT_RUN", "passed": None, "total": None}
    if engineering.get("passed") is not None and engineering.get("total") is not None:
        engineering_rate = _rate(int(engineering["passed"]), int(engineering["total"]))
    else:
        engineering_rate = _rate(0, 0)
    engineering_rate["status"] = engineering.get("status", "NOT_RUN")
    if engineering.get("notes"):
        engineering_rate["notes"] = engineering["notes"]

    failure_counts: dict[str, int] = {}
    scenario_failure_counts: dict[str, int] = {}
    for case in cases:
        if case["result"] not in {"FAIL", "PARTIAL"} or not case["failure_layer"]:
            continue
        target = failure_counts if case["counts_toward_acceptance_metrics"] else scenario_failure_counts
        layer = str(case["failure_layer"])
        target[layer] = target.get(layer, 0) + 1

    return {
        "denominator_rules": {
            "acceptance": "Top-level task records with overall result PASS/FAIL/PARTIAL. NOT_RUN and WAITING_FOR_UAT are excluded; PARTIAL is in the denominator but not the PASS numerator.",
            "owner_uat": "Top-level task records whose owner_uat_result is PASS/FAIL/PARTIAL. NOT_RUN and WAITING_FOR_UAT are excluded; PARTIAL is in the denominator but not the PASS numerator.",
            "wrong_action": "All task/scenario records with action_executed=true and an explicit boolean wrong_action value. Missing/no-action outcomes are not silently counted as wrong-target executions.",
            "engineering_regression": "passed tests / total tests only when a current regression count is explicitly supplied; otherwise N/A.",
        },
        "result_counts": _result_counts(cases, top_level_only=True),
        "scenario_result_counts": _result_counts(cases, top_level_only=False),
        "acceptance_pass_rate": acceptance,
        "core_acceptance_pass_rate": core,
        "p0_acceptance_pass_rate": p0,
        "owner_uat_pass_rate": owner_uat,
        "wrong_action_rate": wrong_action,
        "engineering_regression_pass_rate": engineering_rate,
        "failure_layer_counts": failure_counts,
        "scenario_failure_layer_counts": scenario_failure_counts,
    }


def compute_iteration_delta(
    current_metrics: dict[str, Any],
    previous_run_path: Path | None,
    current_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if previous_run_path is None:
        return {"available": False, "reason": "no previous run supplied"}
    if previous_run_path.suffix.lower() != ".json":
        return {
            "available": False,
            "reason": "previous run is not structured JSON; historical Markdown is not parsed or guessed",
        }
    try:
        previous = json.loads(previous_run_path.read_text(encoding="utf-8"))
        previous_metric = previous["metrics"]["acceptance_pass_rate"]
        previous_counts = previous["metrics"]["result_counts"]
        previous_cases = previous.get("cases", [])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return {"available": False, "reason": f"previous structured run lacks required metrics: {exc}"}
    current_metric = current_metrics["acceptance_pass_rate"]
    if previous_metric.get("rate") is None or current_metric.get("rate") is None:
        return {"available": False, "reason": "current or previous acceptance denominator is zero"}
    current_pass = int(current_metrics["result_counts"]["PASS"])
    previous_pass = int(previous_counts["PASS"])
    current_fail_ids = {
        case["eval_case_id"]
        for case in (current_cases or [])
        if case.get("counts_toward_acceptance_metrics") and case.get("result") == "FAIL"
    }
    previous_fail_ids = {
        case["eval_case_id"]
        for case in previous_cases
        if case.get("counts_toward_acceptance_metrics") and case.get("result") == "FAIL"
    }
    failure_ids_available = current_cases is not None and isinstance(previous_cases, list)
    return {
        "available": True,
        "current_pass_count": current_pass,
        "previous_pass_count": previous_pass,
        "net_pass_delta": current_pass - previous_pass,
        "current_pass_rate": current_metric["rate"],
        "previous_pass_rate": previous_metric["rate"],
        "percentage_point_delta": (current_metric["rate"] - previous_metric["rate"]) * 100.0,
        "fixed_failures_count": len(previous_fail_ids - current_fail_ids) if failure_ids_available else None,
        "new_failures_count": len(current_fail_ids - previous_fail_ids) if failure_ids_available else None,
    }


def load_run_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_run_data(run_data: dict[str, Any], schema: dict[str, Any]) -> None:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(run_data), key=lambda error: list(error.absolute_path))
    if errors:
        details = "; ".join(f"{'/'.join(map(str, err.absolute_path)) or '<root>'}: {err.message}" for err in errors[:8])
        raise ValueError(f"run JSON schema validation failed: {details}")


def _fmt_rate(metric: dict[str, Any]) -> str:
    rate = metric.get("rate")
    if rate is None:
        return "N/A"
    return f"{rate * 100:.2f}% ({metric['numerator']}/{metric['denominator']})"


def _fmt_execution_coverage(run_data: dict[str, Any]) -> str:
    executed = int(run_data["metrics"]["acceptance_pass_rate"]["denominator"])
    total = int(run_data["corpus"]["task_count"])
    if total == 0:
        return "N/A"
    return f"{executed / total * 100:.2f}% ({executed}/{total})"


def render_run_markdown(run_data: dict[str, Any]) -> str:
    meta = run_data["run_metadata"]
    metrics = run_data["metrics"]
    counts = metrics["result_counts"]
    scenarios = metrics["scenario_result_counts"]
    delta = run_data["iteration_delta"]
    lines = [
        f"# {meta['run_id']} — Quantitative Product Eval",
        "",
        "> Machine-generated from the fixed Product Acceptance corpus plus explicit recorded evidence. Missing evidence is never promoted to PASS.",
        "",
        "## Run Identity",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Product version | `{meta['product_version']}` |",
        f"| Git HEAD | `{meta['git_head'] if meta['git_head'] else 'unavailable'}` |",
        f"| Git HEAD source | {meta['git_head_source']} |",
        f"| Worktree state ID | `{meta['worktree_state_id']}` |",
        f"| Eval corpus | `{meta['eval_corpus_version']}` |",
        f"| Platform | `{meta['platform']}` |",
        f"| Python | `{meta['python_version']}` |",
        f"| Provider | `{meta['provider']}` |",
        f"| Model | `{meta['model']}` |",
        f"| Database fixture/state | {meta['database_fixture_or_state_snapshot']} |",
        "",
        "## Quantitative Summary",
        "",
        f"Top-level corpus: **{run_data['corpus']['task_count']} tasks**. Evaluable this run: **{metrics['acceptance_pass_rate']['denominator']}**.",
        f"Execution Coverage: **{_fmt_execution_coverage(run_data)}**.",
        "",
        f"- PASS: **{counts['PASS']}**",
        f"- FAIL: **{counts['FAIL']}**",
        f"- PARTIAL: **{counts['PARTIAL']}**",
        f"- NOT_RUN: **{counts['NOT_RUN']}**",
        f"- WAITING_FOR_UAT: **{counts['WAITING_FOR_UAT']}**",
        f"- Acceptance Pass Rate: **{_fmt_rate(metrics['acceptance_pass_rate'])}**",
        f"- Core Acceptance Pass Rate: **{_fmt_rate(metrics['core_acceptance_pass_rate'])}**",
        f"- P0 Acceptance Pass Rate: **{_fmt_rate(metrics['p0_acceptance_pass_rate'])}**",
        f"- Owner UAT Pass Rate: **{_fmt_rate(metrics['owner_uat_pass_rate'])}**",
        f"- Wrong Action Rate: **{_fmt_rate(metrics['wrong_action_rate'])}**",
        f"- Engineering Regression: **{_fmt_rate(metrics['engineering_regression_pass_rate'])}** ({metrics['engineering_regression_pass_rate']['status']})",
        "",
        "### Denominator Rules",
        "",
    ]
    for key, value in metrics["denominator_rules"].items():
        lines.append(f"- **{key}**: {value}")

    lines.extend(["", "## Top-level Result Matrix", "", "| Case | Type | Priority | Auto | Owner UAT | Overall | Failure layer |", "|---|---|---|---|---|---|---|"])
    for case in run_data["cases"]:
        if case["scope"] != "task":
            continue
        lines.append(
            f"| `{case['eval_case_id']}` | {case['type']} | {case['priority']} | {case['automated_result']} | {case['owner_uat_result']} | **{case['result']}** | {case['failure_layer'] or '—'} |"
        )

    child_cases = [case for case in run_data["cases"] if case["scope"] == "scenario"]
    if child_cases:
        lines.extend([
            "",
            "## Child Scenarios (not added to the 15-task acceptance denominator)",
            "",
            f"Scenario results: PASS {scenarios['PASS']} / FAIL {scenarios['FAIL']} / PARTIAL {scenarios['PARTIAL']} / NOT_RUN {scenarios['NOT_RUN']} / WAITING_FOR_UAT {scenarios['WAITING_FOR_UAT']}.",
            "",
            "| Scenario | Parent | Auto | Owner UAT | Overall | Wrong action | Notes |",
            "|---|---|---|---|---|---|---|",
        ])
        for case in child_cases:
            notes = " ".join(case["notes"]) or "—"
            wrong = "N/A" if case["wrong_action"] is None else str(case["wrong_action"]).lower()
            lines.append(
                f"| `{case['eval_case_id']}` {case['name']} | `{case['parent_task_id']}` | {case['automated_result']} | {case['owner_uat_result']} | **{case['result']}** | {wrong} | {notes} |"
            )

    lines.extend(["", "## Failure Layers", ""])
    if metrics["failure_layer_counts"]:
        for layer, count in sorted(metrics["failure_layer_counts"].items()):
            lines.append(f"- {layer}: {count}")
    else:
        lines.append("- No top-level FAIL/PARTIAL failure layer was recorded in this run.")

    lines.extend(["", "## Iteration Delta", ""])
    if delta.get("available"):
        lines.extend([
            f"- PASS count: {delta['previous_pass_count']} → {delta['current_pass_count']} (net {delta['net_pass_delta']:+d})",
            f"- Pass rate: {delta['previous_pass_rate'] * 100:.2f}% → {delta['current_pass_rate'] * 100:.2f}% ({delta['percentage_point_delta']:+.2f} pp)",
        ])
    else:
        lines.append(f"**Unavailable:** {delta['reason']}")

    lines.extend([
        "",
        "## Engineering Regression",
        "",
        f"Status: **{metrics['engineering_regression_pass_rate']['status']}**.",
    ])
    if metrics["engineering_regression_pass_rate"].get("notes"):
        lines.append(metrics["engineering_regression_pass_rate"]["notes"])

    lines.extend([
        "",
        "## Manual / UAT Boundary",
        "",
        "`Automated Result` never upgrades an Owner-UAT-required case to PASS. `WAITING_FOR_UAT` and `NOT_RUN` remain explicit and are excluded from the Owner UAT denominator.",
        "",
    ])
    return "\n".join(lines)


def render_scorecard_block(run_data: dict[str, Any]) -> str:
    metrics = run_data["metrics"]
    counts = metrics["result_counts"]
    delta = run_data["iteration_delta"]
    delta_text = (
        f"{delta['previous_pass_rate'] * 100:.2f}% → {delta['current_pass_rate'] * 100:.2f}% ({delta['percentage_point_delta']:+.2f} pp)"
        if delta.get("available")
        else f"N/A — {delta['reason']}"
    )
    return "\n".join(
        [
            SCORECARD_START,
            f"## Current Quantitative Snapshot — {run_data['run_metadata']['run_id']}",
            "",
            "> Generated from the machine-readable run JSON. Do not hand-edit arithmetic in this block.",
            "",
            f"- Corpus tasks: **{run_data['corpus']['task_count']}**",
            f"- Execution Coverage: **{_fmt_execution_coverage(run_data)}**",
            f"- PASS / FAIL / PARTIAL / NOT RUN / WAITING: **{counts['PASS']} / {counts['FAIL']} / {counts['PARTIAL']} / {counts['NOT_RUN']} / {counts['WAITING_FOR_UAT']}**",
            f"- Acceptance Pass Rate: **{_fmt_rate(metrics['acceptance_pass_rate'])}**",
            f"- Core Pass Rate: **{_fmt_rate(metrics['core_acceptance_pass_rate'])}**",
            f"- P0 Pass Rate: **{_fmt_rate(metrics['p0_acceptance_pass_rate'])}**",
            f"- Owner UAT Pass Rate: **{_fmt_rate(metrics['owner_uat_pass_rate'])}**",
            f"- Wrong Action Rate: **{_fmt_rate(metrics['wrong_action_rate'])}**",
            f"- Engineering Regression: **{_fmt_rate(metrics['engineering_regression_pass_rate'])}** ({metrics['engineering_regression_pass_rate']['status']})",
            f"- Iteration Delta: **{delta_text}**",
            "",
            "> Acceptance rate is calculated only across executed/evaluable top-level tasks; it is not whole-corpus coverage.",
            "",
            f"Machine-readable source: `eval/runs/{run_data['run_metadata']['run_id']}.json`",
            SCORECARD_END,
        ]
    )


def render_scorecard_history(runs: list[dict[str, Any]]) -> str:
    lines = [
        SCORECARD_HISTORY_START,
        "## Structured Eval History",
        "",
        "> Generated from `eval/runs/P22-EVAL-*.json`. Markdown-only historical runs are intentionally not inferred.",
        "",
        "| Run | Execution Coverage | Acceptance | Owner UAT | Engineering Regression | Fixed Failures | New Failures |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if not runs:
        lines.append("| — | — | — | — | — | — | — |")
    for run in runs:
        metrics = run["metrics"]
        delta = run.get("iteration_delta", {})
        fixed = str(delta.get("fixed_failures_count")) if delta.get("available") and delta.get("fixed_failures_count") is not None else "—"
        new = str(delta.get("new_failures_count")) if delta.get("available") and delta.get("new_failures_count") is not None else "—"
        engineering = metrics["engineering_regression_pass_rate"]
        engineering_text = _fmt_rate(engineering)
        if engineering.get("status"):
            engineering_text += f" ({engineering['status']})"
        lines.append(
            "| `{run}` | **{coverage}** | {acceptance} | {uat} | {engineering} | {fixed} | {new} |".format(
                run=run["run_metadata"]["run_id"],
                coverage=_fmt_execution_coverage(run),
                acceptance=_fmt_rate(metrics["acceptance_pass_rate"]),
                uat=_fmt_rate(metrics["owner_uat_pass_rate"]),
                engineering=engineering_text,
                fixed=fixed,
                new=new,
            )
        )
    lines.extend(["", SCORECARD_HISTORY_END])
    return "\n".join(lines)


def _replace_or_insert_generated_block(text: str, *, start_marker: str, end_marker: str, block: str, insert_before: str | None = None) -> str:
    pattern = re.compile(re.escape(start_marker) + r".*?" + re.escape(end_marker), re.DOTALL)
    if pattern.search(text):
        return pattern.sub(block, text)
    if insert_before and insert_before in text:
        return text.replace(insert_before, block + "\n\n" + insert_before, 1)
    return block + "\n\n" + text


def update_scorecard(
    scorecard_path: Path,
    run_data: dict[str, Any],
    *,
    history_runs: list[dict[str, Any]] | None = None,
) -> None:
    text = scorecard_path.read_text(encoding="utf-8")
    snapshot = render_scorecard_block(run_data)
    history = render_scorecard_history(history_runs or [run_data])
    text = _replace_or_insert_generated_block(
        text,
        start_marker=SCORECARD_START,
        end_marker=SCORECARD_END,
        block=snapshot,
        insert_before=SCORECARD_HISTORY_START if SCORECARD_HISTORY_START in text else "## Metric Definitions",
    )
    text = _replace_or_insert_generated_block(
        text,
        start_marker=SCORECARD_HISTORY_START,
        end_marker=SCORECARD_HISTORY_END,
        block=history,
        insert_before="## Metric Definitions",
    )
    scorecard_path.write_text(text, encoding="utf-8")


def build_run_data(
    *,
    root: Path,
    corpus_path: Path,
    schema_path: Path,
    run_id: str,
    raw_records: list[dict[str, Any]],
    git_head: str | None,
    git_head_source: str,
    provider: str,
    model: str,
    database_state: str,
    previous_run_path: Path | None,
    engineering_regression: dict[str, Any],
    timestamp: str | None = None,
) -> dict[str, Any]:
    corpus = load_corpus_index(corpus_path)
    cases = build_case_records(raw_records, corpus)
    metrics = compute_metrics(cases, engineering_regression=engineering_regression)
    iteration_delta = compute_iteration_delta(metrics, previous_run_path, cases)
    run_data = {
        "schema_version": 1,
        "run_metadata": {
            "run_id": run_id,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "product_version": read_project_version(root / "pyproject.toml"),
            "git_head": git_head,
            "git_head_source": git_head_source,
            "worktree_state_id": compute_worktree_state_id(root, corpus),
            "eval_corpus_version": corpus.eval_corpus_version,
            "platform": platform_module.platform(),
            "python_version": platform_module.python_version(),
            "provider": provider,
            "model": model,
            "database_fixture_or_state_snapshot": database_state,
        },
        "corpus": {
            "path": corpus_path.relative_to(root).as_posix(),
            "schema_version": corpus.schema_version,
            "version": corpus.eval_corpus_version,
            "task_count": len(corpus.tasks),
        },
        "cases": cases,
        "metrics": metrics,
        "iteration_delta": iteration_delta,
    }
    validate_run_data(run_data, load_run_schema(schema_path))
    return run_data


def load_result_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("case_results"), list):
        raise ValueError("results file must be an object containing case_results[]")
    engineering = raw.get("engineering_regression") or {"status": "NOT_RUN", "passed": None, "total": None}
    return raw["case_results"], engineering
