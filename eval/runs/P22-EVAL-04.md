# P22-EVAL-04 — Quantitative Product Eval

> Machine-generated from the fixed Product Acceptance corpus plus explicit recorded evidence. Missing evidence is never promoted to PASS.

## Run Identity

| Field | Value |
|---|---|
| Product version | `0.1.0` |
| Git HEAD | `ec55a033e7f51d9869a889c19d95c913c475ba53` |
| Git HEAD source | detected from repository |
| Worktree state ID | `sha256-manifest-v1:f36d3b1687b9eac333d1510b76a5db348adac6c1e62bc2d123e3bf7e05ac5467` |
| Eval corpus | `p22-product-acceptance-v1` |
| Platform | `macOS-26.6.2-arm64-arm-64bit` |
| Python | `3.12.13` |
| Provider | `N/A` |
| Model | `N/A` |
| Database fixture/state | unavailable — quantitative aggregation only; no live database fixture was executed |

## Quantitative Summary

Top-level corpus: **15 tasks**. Evaluable this run: **2**.

- PASS: **1**
- FAIL: **1**
- PARTIAL: **0**
- NOT_RUN: **13**
- WAITING_FOR_UAT: **0**
- Acceptance Pass Rate: **50.00% (1/2)**
- Core Acceptance Pass Rate: **N/A**
- P0 Acceptance Pass Rate: **50.00% (1/2)**
- Owner UAT Pass Rate: **50.00% (1/2)**
- Wrong Action Rate: **0.00% (0/4)**
- Engineering Regression: **N/A** (NOT_RUN)

### Denominator Rules

- **acceptance**: Top-level task records with overall result PASS/FAIL/PARTIAL. NOT_RUN and WAITING_FOR_UAT are excluded; PARTIAL is in the denominator but not the PASS numerator.
- **owner_uat**: Top-level task records whose owner_uat_result is PASS/FAIL/PARTIAL. NOT_RUN and WAITING_FOR_UAT are excluded; PARTIAL is in the denominator but not the PASS numerator.
- **wrong_action**: All task/scenario records with action_executed=true and an explicit boolean wrong_action value. Missing/no-action outcomes are not silently counted as wrong-target executions.
- **engineering_regression**: passed tests / total tests only when a current regression count is explicitly supplied; otherwise N/A.

## Top-level Result Matrix

| Case | Type | Priority | Auto | Owner UAT | Overall | Failure layer |
|---|---|---|---|---|---|---|
| `ACPT-REC-001` | Core | P0 | NOT_RUN | NOT_RUN | **NOT_RUN** | — |
| `ACPT-REC-002` | Core | P0 | NOT_RUN | NOT_RUN | **NOT_RUN** | — |
| `ACPT-REC-003` | Core | P0 | NOT_RUN | NOT_RUN | **NOT_RUN** | — |
| `ACPT-DIS-001` | Core | P0 | NOT_RUN | NOT_RUN | **NOT_RUN** | — |
| `ACPT-CTX-001` | Core | P0 | NOT_RUN | NOT_RUN | **NOT_RUN** | — |
| `ACPT-CTX-002` | Core | P0 | NOT_RUN | NOT_RUN | **NOT_RUN** | — |
| `ACPT-CTX-003` | Core | P0 | NOT_RUN | NOT_RUN | **NOT_RUN** | — |
| `ACPT-FAIL-001` | Core | P0 | NOT_RUN | NOT_RUN | **NOT_RUN** | — |
| `ACPT-PRE-001` | Core | P0 | NOT_RUN | NOT_RUN | **NOT_RUN** | — |
| `ACPT-PLAY-001` | Core | P0 | NOT_RUN | NOT_RUN | **NOT_RUN** | — |
| `ACPT-FAIL-002` | Core | P0 | NOT_RUN | NOT_RUN | **NOT_RUN** | — |
| `ACPT-COLD-001` | Core | P1 | NOT_RUN | NOT_RUN | **NOT_RUN** | — |
| `ACPT-FB-001` | Core | P0 | NOT_RUN | NOT_RUN | **NOT_RUN** | — |
| `ACPT-P22-ROUTE-001` | Extension | P0 | PASS | PASS | **PASS** | — |
| `ACPT-P22-CTX-004` | Extension | P0 | PARTIAL | FAIL | **FAIL** | Authority |

## Child Scenarios (not added to the 15-task acceptance denominator)

Scenario results: PASS 2 / FAIL 1 / PARTIAL 0 / NOT_RUN 0 / WAITING_FOR_UAT 3.

| Scenario | Parent | Auto | Owner UAT | Overall | Wrong action | Notes |
|---|---|---|---|---|---|---|
| `ACPT-P22-CTX-004-A` Named target acceptance | `ACPT-P22-CTX-004` | PASS | PASS | **PASS** | false | Exact target identity is preserved across the pending offer continuation. |
| `ACPT-P22-CTX-004-B` Acceptance synonym | `ACPT-P22-CTX-004` | PASS | WAITING_FOR_UAT | **WAITING_FOR_UAT** | false | Automated evidence exists for 好的/可以/试听吧 semantics; no current Owner UAT result is claimed. |
| `ACPT-P22-CTX-004-C` Decline | `ACPT-P22-CTX-004` | PASS | WAITING_FOR_UAT | **WAITING_FOR_UAT** | N/A | Automated decline/consume behavior is covered; current Owner UAT is not recorded. |
| `ACPT-P22-CTX-004-D` Decline plus explicit override | `ACPT-P22-CTX-004` | PASS | PASS | **PASS** | false | The old pending offer is not consumed as a pure decline when a substantive new request is present. |
| `ACPT-P22-CTX-004-E` Chained explicit-index Preview offer | `ACPT-P22-CTX-004` | NOT_RUN | FAIL | **FAIL** | false | No wrong target executed; failure is missing structured OfferedAction production for this chain. |
| `ACPT-P22-CTX-004-F` Consumed offer replay protection | `ACPT-P22-CTX-004` | PASS | WAITING_FOR_UAT | **WAITING_FOR_UAT** | false | Automated consume-once behavior is verified; no current Owner UAT result is claimed. |

## Failure Layers

- Authority: 1

## Iteration Delta

**Unavailable:** previous run is not structured JSON; historical Markdown is not parsed or guessed

## Engineering Regression

Status: **NOT_RUN**.
P22-EVAL-04 records quantitative Product Eval evidence only; no current full engineering regression was executed in this run.

## Manual / UAT Boundary

`Automated Result` never upgrades an Owner-UAT-required case to PASS. `WAITING_FOR_UAT` and `NOT_RUN` remain explicit and are excluded from the Owner UAT denominator.
