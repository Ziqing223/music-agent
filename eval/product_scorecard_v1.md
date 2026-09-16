# Music Agent Product Eval Scorecard V1

> Aggregated view for the fixed `p22-product-acceptance-v1` corpus. Quantitative current/history sections are generator-owned and come only from structured run JSON. Engineering verification and Owner UAT remain distinct evidence sources.

<!-- PRODUCT_EVAL_GENERATED_START -->
## Current Quantitative Snapshot — P22-EVAL-05

> Generated from the machine-readable run JSON. Do not hand-edit arithmetic in this block.

- Corpus tasks: **15**
- Execution Coverage: **13.33% (2/15)**
- PASS / FAIL / PARTIAL / NOT RUN / WAITING: **2 / 0 / 0 / 13 / 0**
- Acceptance Pass Rate: **100.00% (2/2)**
- Core Pass Rate: **N/A**
- P0 Pass Rate: **100.00% (2/2)**
- Owner UAT Pass Rate: **100.00% (2/2)**
- Wrong Action Rate: **0.00% (0/6)**
- Engineering Regression: **100.00% (5421/5421)** (PASS)
- Iteration Delta: **50.00% → 100.00% (+50.00 pp)**

> Acceptance rate is calculated only across executed/evaluable top-level tasks; it is not whole-corpus coverage.

Machine-readable source: `eval/runs/P22-EVAL-05.json`
<!-- PRODUCT_EVAL_GENERATED_END -->

<!-- PRODUCT_EVAL_HISTORY_START -->
## Structured Eval History

> Generated from `eval/runs/P22-EVAL-*.json`. Markdown-only historical runs are intentionally not inferred.

| Run | Execution Coverage | Acceptance | Owner UAT | Engineering Regression | Fixed Failures | New Failures |
|---|---:|---:|---:|---:|---:|---:|
| `P22-EVAL-04` | **13.33% (2/15)** | 50.00% (1/2) | 50.00% (1/2) | N/A (NOT_RUN) | — | — |
| `P22-EVAL-05` | **13.33% (2/15)** | 100.00% (2/2) | 100.00% (2/2) | 100.00% (5421/5421) (PASS) | 1 | 0 |

<!-- PRODUCT_EVAL_HISTORY_END -->

## Metric Definitions

The scorecard separates **execution coverage** from **acceptance among executed/evaluable tasks**. For example, `2/2 PASS` with `2/15` execution coverage means 100% acceptance on the executed slice, not 100% coverage of the corpus.

**Acceptance Pass Rate**

```text
PASS top-level Product Eval tasks
/
all executed/evaluable top-level tasks (PASS / FAIL / PARTIAL)
```

`NOT_RUN` and `WAITING_FOR_UAT` are excluded from that denominator. `PARTIAL` is included in the denominator but not in the PASS numerator.

**Owner UAT Pass Rate**

```text
Owner-UAT PASS top-level tasks
/
top-level tasks with explicit Owner UAT PASS / FAIL / PARTIAL
```

**Wrong Action Rate** — count a wrong action when any of the following occurs:

- selected target is not the intended target;
- `preview_only` is upgraded to formal play;
- an unauthorized action executes;
- intended / selected / executed / readback / user-visible verified target diverge while the system still claims success;
- the user requests target A and the system executes target B.

**Engineering Regression Pass Rate** is `passed tests / total tests` only when a current explicit regression count is supplied. It is not a Product Acceptance rate.

## Evidence Contract

For every formal Product Acceptance execution, capture or explicitly mark unavailable:

```text
eval_case_id
product_version
git_head
worktree_state_id
eval_corpus_version
platform
python_version
provider
model
database_fixture_or_state_snapshot
run_id
turn_id
input
precondition
expected_behavior
actual_intent
actual_route
relevant_context
selected_target
selection_authority
selection_grant
tool_name
tool_parameters
tool_result
preview_or_playback_eligibility
executed_target
readback_target
final_state
final_response
retry_count
latency_ms
automated_result
owner_uat_result
failure_layer
root_cause_status
```

Unavailable evidence stays `unavailable`; do not infer a successful real action from missing readback.

For real actions, success requires all applicable identities to reconcile:

```text
intended target
== selected target
== executed target
== readback target
== user-visible verified target
```

## Failure Classification

Use exactly one primary failure layer, with optional secondary layers:

```text
Intent
Context
Authority
Selection
Tool
Execution
Readback
Presentation
Observability
Eval Infrastructure
```

A failed UAT remains `Unverified` until runtime/journal/readback evidence confirms root cause. Only then should a long-lived regression candidate be created.

## UAT Failure → Regression Candidate

```text
Owner UAT FAIL
→ save Evidence Snapshot
→ classify Failure Layer
→ form Root Cause Hypothesis
→ runtime / journal / readback audit
→ Root Cause Confirmed?
    No  → remain Unverified
    Yes → check Product Contract
        → add the smallest representative targeted/regression case only if needed
        → minimal implementation
        → targeted/domain/full regression
        → Owner UAT when applicable
        → keep as Active Regression only if representative long term
```

## Coverage Expansion

The generated snapshot is authoritative for current `PASS / FAIL / PARTIAL / NOT_RUN / WAITING_FOR_UAT` counts. Remaining `NOT_RUN` top-level cases are coverage backlog; they are not silently interpreted as pass or fail. Expand execution coverage with explicit evidence rather than rewriting historical runs.

## Run Status Vocabulary

Keep project verification semantics unchanged:

```text
Discussed
Proposed
Decided
Implemented
Automated Verified
Owner UAT Verified
Unverified
Rejected / Deferred
```

A Product Eval scorecard must not convert `Automated Verified` into `Owner UAT Verified` merely because engineering tests pass.
