"""P15 burn-down Issue 1: repair model-fabricated ``produced_at`` timestamps in
durable recommendation history, rebuilt only from the journal's authoritative
``completed_at`` instants (see ``music_agent.durable_produced_at_repair``).

Dry run by default -- prints the exact repair plan (runs, catalog resets,
blocked rows, within-tolerance rows) and touches nothing. ``--apply`` first
writes a sqlite backup next to the database (never overwriting an existing
one), then repairs inside one exclusive transaction with the
recommendation_runs immutability triggers suspended only for its duration,
then re-runs the read-only plan to prove the drift is gone.

  PYTHONPATH=src .venv/bin/python tools/repair_durable_produced_at.py \\
      --db ~/MusicAgent/music_agent.db
  PYTHONPATH=src .venv/bin/python tools/repair_durable_produced_at.py \\
      --db ~/MusicAgent/music_agent.db --apply
  PYTHONPATH=src .venv/bin/python tools/repair_durable_produced_at.py \\
      --db ~/MusicAgent/music_agent.db --apply --tolerance-hours 0.5
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from music_agent.durable_produced_at_repair import (
    apply_repair,
    build_repair_plan,
)

DEFAULT_DATABASE_PATH = Path.home() / "MusicAgent" / "music_agent.db"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DATABASE_PATH, metavar="PATH"
    )
    parser.add_argument(
        "--tolerance-hours",
        type=float,
        default=0.25,
        help="drift below this (in hours) is not repaired (default 0.25)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the backup and apply the plan (default: dry run only)",
    )
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"database not found: {args.db}", file=sys.stderr)
        return 1

    plan = build_repair_plan(args.db, tolerance_hours=args.tolerance_hours)
    skip_reasons = Counter(reason for _, reason, _ in plan.catalog_skipped)
    print(f"database:              {plan.database_path}")
    print(f"tolerance:             {plan.tolerance_hours}h")
    print(f"total runs:            {plan.total_runs}")
    print(f"within tolerance:      {len(plan.within_tolerance)}")
    print(f"drifted, repairable:   {len(plan.entries)}")
    print(f"drifted, blocked:      {len(plan.blocked)}")
    print(f"catalog resets:        {len(plan.catalog_resets)}")
    print(
        "catalog skipped:       "
        + (", ".join(f"{reason} x{n}" for reason, n in sorted(skip_reasons.items())) or "0")
    )
    for entry in plan.entries:
        print(
            f"  run {entry.run_id}: {entry.old_produced_at} -> "
            f"{entry.new_produced_at} (journal {entry.journal_request_id})"
        )
    for blocked in plan.blocked:
        print(
            f"  BLOCKED run {blocked.run_id} [{blocked.reason}, "
            f"refs={blocked.journal_ref_count}]: {blocked.old_produced_at}"
        )
    for reset in plan.catalog_resets:
        print(
            f"  catalog {reset.canonical_id}: "
            f"first {reset.old_first} -> {reset.new_first}, "
            f"last {reset.old_last} -> {reset.new_last}"
        )
    if not plan.has_work:
        print("nothing to repair")
        return 0
    if not args.apply:
        print("dry run complete -- rerun with --apply to repair")
        return 0

    report = apply_repair(args.db, plan)
    print(f"applied: {report}")
    verdict = build_repair_plan(args.db, tolerance_hours=args.tolerance_hours)
    remaining = len(verdict.entries) + len(verdict.blocked)
    print(
        f"post-apply verification: {verdict.total_runs} runs, "
        f"{len(verdict.entries)} repairable, {len(verdict.blocked)} blocked"
    )
    if len(verdict.entries) != 0:
        print("VERIFICATION FAILED: repairable drift remains", file=sys.stderr)
        return 1
    print(
        "VERIFICATION OK: no repairable drift remains "
        f"({remaining - len(verdict.entries)} rows beyond rebuildable proof, "
        "left untouched)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())