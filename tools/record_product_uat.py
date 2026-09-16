#!/usr/bin/env python3
"""Record explicit Owner UAT evidence into a Product Eval results JSON file.

This tool is deterministic local infrastructure. It never changes automated_result
and never infers a PASS/FAIL from prose or model output.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import sys
from typing import Any

from product_eval_runner import OWNER_UAT_RESULT_VALUES


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode if path.exists() else None
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temp_path, mode)
        os.replace(temp_path, path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise


def record_owner_uat(
    results_file: Path,
    *,
    case_id: str,
    result: str,
    notes: str | None = None,
    reference: str = "Owner UAT",
) -> dict[str, Any]:
    result = result.strip().upper()
    if result not in OWNER_UAT_RESULT_VALUES:
        allowed = ", ".join(sorted(OWNER_UAT_RESULT_VALUES))
        raise ValueError(f"invalid Owner UAT result {result!r}; expected one of: {allowed}")

    data = json.loads(results_file.read_text(encoding="utf-8"))
    records = data.get("case_results") if isinstance(data, dict) else None
    if not isinstance(records, list):
        raise ValueError("results file must be an object containing case_results[]")

    matches = [record for record in records if record.get("eval_case_id") == case_id]
    if len(matches) != 1:
        if not matches:
            raise ValueError(f"unknown eval_case_id: {case_id}")
        raise ValueError(f"duplicate eval_case_id in results file: {case_id}")

    record = matches[0]
    automated_before = record.get("automated_result")
    record["owner_uat_result"] = result

    evidence = record.setdefault("evidence", [])
    if not isinstance(evidence, list):
        raise ValueError(f"{case_id}: evidence must be a list")
    item: dict[str, Any] = {
        "kind": "owner_uat",
        "reference": reference,
    }
    if notes:
        item["notes"] = notes
    evidence.append(item)

    if record.get("automated_result") != automated_before:
        raise AssertionError("record_owner_uat must not modify automated_result")

    _atomic_write_json(results_file, data)
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record explicit Owner UAT evidence for one Eval case.")
    parser.add_argument("--results-file", required=True, type=Path)
    parser.add_argument("--case", required=True, dest="case_id")
    parser.add_argument("--result", required=True)
    parser.add_argument("--notes")
    parser.add_argument("--reference", default="Owner UAT")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    record = record_owner_uat(
        args.results_file.resolve(),
        case_id=args.case_id,
        result=args.result,
        notes=args.notes,
        reference=args.reference,
    )
    print(f"results_file={args.results_file.resolve()}")
    print(f"case={record['eval_case_id']}")
    print(f"owner_uat_result={record['owner_uat_result']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
