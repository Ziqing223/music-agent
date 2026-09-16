#!/usr/bin/env python3
"""CLI for recording and aggregating the fixed Product Acceptance corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from product_eval_runner import (
    build_run_data,
    detect_git_head,
    load_result_records,
    load_structured_run_history,
    render_run_markdown,
    resolve_previous_run_path,
    update_scorecard,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record and aggregate a quantitative Product Eval run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-file", required=True, type=Path, help="Explicit case/UAT evidence JSON; no result is inferred from Markdown.")
    parser.add_argument(
        "--previous-run",
        help=(
            "Previous structured JSON run path, or 'auto'. If omitted, the nearest "
            "earlier eval/runs/P22-EVAL-*.json run is auto-discovered."
        ),
    )
    parser.add_argument("--git-head", help="Externally supplied Git HEAD when .git is unavailable.")
    parser.add_argument("--provider", default="N/A")
    parser.add_argument("--model", default="N/A")
    parser.add_argument("--database-state", default="unavailable — no database/state snapshot supplied for this aggregation run")
    parser.add_argument("--timestamp", help="Optional ISO-8601 timestamp override (useful for reproducible fixtures).")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--scorecard", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--update-scorecard", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    corpus = (args.corpus or root / "eval/product_acceptance_v1.yaml").resolve()
    schema = (args.schema or root / "eval/product_eval_run.schema.json").resolve()
    scorecard = (args.scorecard or root / "eval/product_scorecard_v1.md").resolve()
    output_json = (args.output_json or root / f"eval/runs/{args.run_id}.json").resolve()
    output_md = (args.output_md or root / f"eval/runs/{args.run_id}.md").resolve()

    records, engineering = load_result_records(args.results_file.resolve())
    if args.git_head:
        git_head = args.git_head
        git_head_source = "externally supplied because uploaded snapshot may not contain .git metadata"
    else:
        git_head, git_head_source = detect_git_head(root)

    previous_run = resolve_previous_run_path(
        args.previous_run,
        runs_dir=root / "eval/runs",
        current_run_id=args.run_id,
    )
    run_data = build_run_data(
        root=root,
        corpus_path=corpus,
        schema_path=schema,
        run_id=args.run_id,
        raw_records=records,
        git_head=git_head,
        git_head_source=git_head_source,
        provider=args.provider,
        model=args.model,
        database_state=args.database_state,
        previous_run_path=previous_run,
        engineering_regression=engineering,
        timestamp=args.timestamp,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(run_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_run_markdown(run_data), encoding="utf-8")
    if args.update_scorecard:
        history_runs = load_structured_run_history(root / "eval/runs", current_run=run_data)
        update_scorecard(scorecard, run_data, history_runs=history_runs)

    print(f"run_id={args.run_id}")
    print(f"json={output_json}")
    print(f"markdown={output_md}")
    print(f"worktree_state_id={run_data['run_metadata']['worktree_state_id']}")
    print(f"acceptance_pass_rate={run_data['metrics']['acceptance_pass_rate']['rate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
