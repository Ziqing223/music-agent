# P22-EVAL-03 — First Product Acceptance Run

> Product Acceptance execution over the fixed `p22-product-acceptance-v1` corpus.
> This run aggregates the existing engineering verification backbone plus controlled runtime evidence.
> It does not modify production code, tests, or the Acceptance contract.

## A. Eval Run Identity

| Field | Value | Notes |
|---|---|---|
| Eval run | `P22-EVAL-03-FIRST-RUN` | First execution of the fixed Product Acceptance corpus |
| Product version | `0.1.0` | `pyproject.toml` / package version |
| Git HEAD | `ec55a033e7f51d9869a889c19d95c913c475ba53` | HEAD alone is not the product identity |
| Branch | `main` | Current uploaded EVAL-02 snapshot |
| Worktree state ID | `sha256-manifest-v1:bdae579fed099fbc98a2d169ca0420c81f9876a97d8d6e5c004e35b3da0698fa` | Recomputed with the corpus algorithm, excluding `.DS_Store`, caches/build outputs, `.git/**`, and `eval/**` |
| Eval corpus version | `p22-product-acceptance-v1` | `eval/product_acceptance_v1.yaml` |
| Platform | `Linux x86_64 (6.18.44)` | Eval execution environment; **not** the Owner's real macOS product host |
| Python version | `3.13.5` | Current eval environment |
| Provider | `N/A` | Acceptance evidence used deterministic/unit/integration fixtures and controlled fake providers; no live external Provider was invoked |
| Model | `N/A` | No live model used for this run |
| Database fixture / state snapshot | `N/A — ephemeral per-test fixtures` | Tests create controlled temporary state; there is no single durable acceptance DB snapshot |
| Engineering regression baseline | `5374 / 5374 OK` | Owner-verified real macOS baseline supplied for P22-S1.11; **not rerun** in this Product Acceptance execution |
| Preflight `git diff --check` | `PASS` | No whitespace/conflict errors |
| Preflight staged state | `0` | No staged files |

### Execution notes

The corpus contains 63 test references, representing 60 unique existing tests. All 60 unique mapped tests were rerun in this eval environment and passed.

Three additional existing tests were run only to close the four-command evidence check for `ACPT-P22-ROUTE-001`:

- `tests.test_intent_router.IntentRouterTest.test_chinese_commands_route_exactly`
- `tests.test_intent_router.SessionAwareRoutingTest.test_other_commands_are_untouched_by_the_session_state`
- `tests.test_playback_tools.PlaybackToolsTest.test_play_pause_next_previous_commands`

All three passed.

No Product Acceptance result below is marked PASS merely because its mapped tests passed. The overall outcome also considers coverage completeness, current runtime evidence, readback requirements, and current-run Owner UAT requirements.

## B. 15-Task Result Matrix

Allowed outcomes used by this run: `PASS`, `FAIL`, `PARTIAL-EVIDENCE`, `NOT-RUN`, `UAT-REQUIRED`.

| Task ID | Automated evidence this run | Runtime / readback evidence | Owner UAT | Overall Result | Failure / evidence layer | Root cause status |
|---|---|---|---|---|---|---|
| `ACPT-REC-001` Scene-based Recommendation | `4/4 PASS` | Scene routing/semantics covered; no current case-level final semantic-fit snapshot against unrelated preference | Conditional; not executed | **PARTIAL-EVIDENCE** | Context / Presentation / Eval Infrastructure | `confirmed` evidence gap, not a confirmed product failure |
| `ACPT-REC-002` Mood-based Recommendation | `3/3 PASS` | Mood/direction routing covered; no current before/after durable-preference evidence proving non-persistence | Conditional; not executed | **PARTIAL-EVIDENCE** | Intent / Context / Eval Infrastructure | `confirmed` evidence gap, not a confirmed product failure |
| `ACPT-REC-003` Similarity | `6/6 PASS` | Strict seed, seed exclusion, similarity-vs-preference and run lifecycle are directly asserted by existing tests | Conditional; not required to establish this controlled contract | **PASS** | — | — |
| `ACPT-DIS-001` Discovery | `4/4 PASS` | End-to-end controlled discovery test asserts actual discovered source, fresh item truth and deterministic final wording | Sample/conditional; not required for this controlled contract | **PASS** | — | — |
| `ACPT-CTX-001` Current Intent vs Long-term Preference | `4/4 PASS` | Similarity domain proves current seed outranks stronger global preference; no general scene/mood priority case | Sample/conditional; not executed | **PARTIAL-EVIDENCE** | Context / Observability | `confirmed` coverage gap |
| `ACPT-CTX-002` Continuous Adjustment | `5/5 PASS` | Active batch, run replacement, verified selection reset and deterministic choose-another are covered | **Required; only historical UAT exists** | **UAT-REQUIRED** | Observability / Eval Infrastructure | `confirmed` current-run evidence gap |
| `ACPT-CTX-003` Ambiguous Context | `4/4 PASS` | Controlled ambiguity reaches clarification/fail-closed; unauthorized action is not executed | Not required | **PASS** | — | — |
| `ACPT-FAIL-001` No Candidate | `4/4 PASS` | Empty direct/inferred generation, no-history-pollution and honest Web fallback are asserted | Not required | **PASS** | — | — |
| `ACPT-PRE-001` Preview Unavailable | `4/4 PASS` | Preview unavailable / no URL / `started=false` fail-closed behavior covered in automation | **Required; no current-run real Preview-unavailable UAT** | **UAT-REQUIRED** | Readback / Eval Infrastructure | `confirmed` current-run UAT gap |
| `ACPT-PLAY-001` Playback Permission Failure | `5/5 PASS` | Permission denial, unavailable adapter and strict readback rejection are covered in automation | **Required; no current-run permission-failure Music.app UAT** | **UAT-REQUIRED** | Readback / Eval Infrastructure | `confirmed` current-run UAT gap |
| `ACPT-FAIL-002` Tool / External Service Failure | `4/4 PASS` | Typed Provider timeout/error and bounded Provider recovery are directly asserted | Conditional; controlled failure evidence is sufficient here | **PASS** | — | — |
| `ACPT-COLD-001` Cold Start / No Preference Data | `4/4 PASS` | Neutral context, absent-confidence and inferred/fresh fallback are covered separately; no single product E2E snapshot joins them | Not required | **PARTIAL-EVIDENCE** | Context / Presentation / Eval Infrastructure | `confirmed` evidence gap |
| `ACPT-FB-001` Feedback Semantics | `5/5 PASS` | Explicit/implicit classification, `Skip != Dislike`, `Complete != Like`, durable explicit application/readback are asserted | Conditional; controlled persistence evidence is sufficient here | **PASS** | — | — |
| `ACPT-P22-ROUTE-001` Deterministic Playback Command | `5/5 mapped PASS` + `3/3 supplemental PASS` | **Current WebShell controlled runtime fails the product contract for continue/next/previous; pause alone is direct** | Conditional; no current real latency UAT | **FAIL** | Execution; latency remains Observability | `confirmed` for route failure; latency causality not measured |
| `ACPT-P22-CTX-004` Conversational Continuation / Pending Action | `2/2 mapped PASS` but they cover only specialized suspension continuation | **Current two-turn WebShell runtime loses the assistant offer/target and executes no target-bound action** | Current known gap reproduced in controlled runtime | **FAIL** | Context / Authority / Selection | `confirmed` |

## C. Core Acceptance Results

Core corpus outcomes:

| Outcome | Count | Tasks |
|---|---:|---|
| PASS | 6 | `REC-003`, `DIS-001`, `CTX-003`, `FAIL-001`, `FAIL-002`, `FB-001` |
| PARTIAL-EVIDENCE | 4 | `REC-001`, `REC-002`, `CTX-001`, `COLD-001` |
| UAT-REQUIRED | 3 | `CTX-002`, `PRE-001`, `PLAY-001` |
| FAIL | 0 | — |
| NOT-RUN | 0 | — |

Strict Core Acceptance Pass Rate for this run:

```text
6 / 13 = 46.15%
```

This is **not** a 53.85% product failure rate. The seven non-PASS Core tasks are evidence/UAT incompleteness: four `PARTIAL-EVIDENCE`, three `UAT-REQUIRED`, and zero confirmed Core behavior failures.

P0 Core tasks exclude only `ACPT-COLD-001` (P1):

```text
P0 strict PASS = 6 / 12 = 50.00%
```

Again, the non-PASS P0 cases are evidence/UAT states, not six product failures.

## D. Extension Results

### ACPT-P22-ROUTE-001 — FAIL

The deterministic router itself recognizes the four closed commands:

```text
继续播放 -> play
暂停     -> pause
下一首   -> next_track
上一首   -> previous_track
```

Existing router/tool tests pass for all four.

However the actual Web App boundary does **not** give all four the same deterministic execution path. A controlled current-code WebShell run sent the exact commands through `/api/chat` using the normal `WebShellApp` + `ProviderAgentLoop` fixture:

| Input | HTTP | Reply | Provider path | Playback adapter action | Product contract |
|---|---:|---|---|---|---|
| `继续播放` | 200 | `好的。` | **ProviderAgentLoop invoked** | none | **FAIL** |
| `暂停` | 200 | `已暂停播放。` | deterministic fast path | `pause` | PASS |
| `下一首` | 200 | `好的。` | **ProviderAgentLoop invoked** | none | **FAIL** |
| `上一首` | 200 | `好的。` | **ProviderAgentLoop invoked** | none | **FAIL** |

The implementation evidence explains the divergence: `_stop_family_fast_path()` executes `pause` / `stop_preview`, but explicitly returns `None` for other routed values, after which `_run_chat()` proceeds to `ProviderAgentLoop.run()`.

This violates the Extension contract:

```text
deterministic closed command
→ Provider rounds = 0
→ direct action
→ required readback
```

for `继续播放`, `下一首`, and `上一首`.

**Latency:** local fixture timings were observed during the controlled run, but the Provider was a local fake. They are not valid product latency measurements and are therefore excluded from P50/P95. The confirmed Provider-path divergence is consistent with the Owner-observed “继续播放响应时间偏长”, but this run does **not** establish that Provider latency is the sole latency root cause.

Failure layer: `Execution`.

Root cause status:

- deterministic-route execution divergence: `confirmed`;
- real-product latency root cause: `unknown` / not measured in this run.

### ACPT-P22-CTX-004 — FAIL

Two controlled two-turn WebShell scenarios were executed with a recording Provider.

#### Offered Preview

Turn 1 Provider reply:

```text
需要我试听 Wendy 吗？
```

Turn 2 user:

```text
开始试听
```

Observed second Provider input:

```text
[("user", "开始试听")]
```

The previous assistant offer and the offered target `Wendy` were absent. No playback/preview adapter action executed.

The deterministic parser also reports:

```text
route_intent("开始试听") -> None
TurnPlan.primary -> unknown
semantic_source -> unresolved
```

#### Offered Resume

Turn 1 Provider reply:

```text
需要继续播放吗？
```

Turn 2 user:

```text
继续
```

Observed second Provider input again contained only:

```text
[("user", "继续")]
```

and no pending assistant-offer context. No playback adapter action executed in this offered-action fixture.

The specialized `AudioSuspension` continuation path still has passing engineering tests; that is not a general conversational offered-action contract.

Current code search confirms that durable `PendingIntent` belongs to write-safety execution and no general target-bound assistant-offer state exists.

Failure layers:

```text
Context
Authority
Selection
```

Root cause status: `confirmed`.

## E. Wrong Action Findings

No wrong-target execution was observed in this Product Acceptance run.

The two Extension failures were **missing/wrong routing or lost continuation context**, not execution of an incorrect target:

- deterministic command failure: no direct action was executed for continue/next/previous in the controlled WebShell run;
- conversational continuation failure: no target-bound preview/resume action was executed.

The Product Eval `Wrong Action Rate` remains `N/A`, because this run did not execute a current real Music.app action set with a complete:

```text
intended
==
selected
==
executed
==
readback
==
user-visible verified target
```

chain.

Historical real-machine playback evidence is not substituted for current-run action identity evidence.

## F. Fail-closed Findings

Current automated acceptance evidence passed for:

- ambiguous context;
- no candidate / empty recommendation;
- Provider/tool timeout and error;
- Preview unavailable / `started=false`;
- playback permission / adapter unavailable;
- player canonical readback mismatch.

No false-success regression was observed in those controlled tests.

`Fail-closed Correctness` is kept `N/A` at Product-scorecard level because the current run lacks current real UAT for the Preview and formal-playback failure cases and therefore does not have a complete product-level denominator.

## G. Clarification Findings

`ACPT-CTX-003` passed: ambiguity produces clarification/fail-closed without unauthorized execution.

A distinct unnecessary-clarification/continuation defect is confirmed in `ACPT-P22-CTX-004`: when an assistant offers a target-bound action, the following short acceptance does not receive that offer/target as authoritative context. A recording Provider therefore receives only the second utterance and can require information the preceding turn had already established.

The global `Unnecessary Clarification Rate` remains `N/A`: one confirmed Extension failure is not a sufficient corpus-wide denominator.

## H. Latency Findings

No valid product P50/P95 latency sample was collected.

The Extension route probe used a local fake Provider specifically to determine route/authority behavior. Its elapsed milliseconds are **not** representative of the real Provider, Music.app, network, or Owner machine and are therefore not used as Product Eval latency.

Current result:

```text
P50 Latency = N/A
P95 Latency = N/A
```

Evidence gap:

```text
No current eval_case_id-keyed real App timing for the four deterministic playback commands.
```

The route audit does establish that `继续播放`, `下一首`, and `上一首` currently enter a Provider round at the WebShell boundary, while `暂停` does not.

## I. UAT-Required Items

The following Core cases require a current-run Owner UAT before they can become PASS:

1. `ACPT-CTX-002` — Continuous Adjustment
   - Automated authority/run-lineage evidence passes.
   - Existing real UAT is historical, not keyed to this eval run.

2. `ACPT-PRE-001` — Preview Unavailable
   - Automated unavailable/started-gate evidence passes.
   - Need a current real Preview-unavailable case proving user-visible state and runtime state stay consistent.

3. `ACPT-PLAY-001` — Playback Permission Failure
   - Automated permission/readback evidence passes.
   - Need a current real Music.app capability/permission failure case proving no false playback-success message.

Historical P20 Owner/system UAT remains useful background evidence for successful Preview restore, formal playback and similarity, but it is not relabeled as this run's UAT.

## J. Evidence Gaps

Confirmed current-run evidence gaps:

- no durable `eval_case_id` propagated into production runtime/journal;
- no unified `turn_id`;
- TurnPlan / actual intent is mostly in-process;
- no single acceptance DB/state snapshot; mapped tests use ephemeral fixtures;
- no current real Provider/model invocation for this run;
- no current real Music.app UAT for three required Core tasks;
- no unified final-response-to-journal join;
- no product-valid deterministic-command latency dataset;
- scene final semantic-fit vs unrelated preference is not one fixed assertion;
- mood non-persistence is not one fixed Product Acceptance assertion;
- cold-start neutral state + explicit intent + final response is not one joined Product Acceptance snapshot;
- no general target-bound conversational offered-action state.

These are evidence/infrastructure gaps unless a task is explicitly marked FAIL above.

## K. Confirmed Failures

Two Product Acceptance failures are confirmed.

### 1. ACPT-P22-ROUTE-001

**Fact:** WebShell directly executes `暂停`, but `继续播放 / 下一首 / 上一首` proceed into `ProviderAgentLoop`.

**Failure layer:** Execution.

**Root cause status:** `confirmed` for the route divergence.

**Do not infer:** the exact share of live latency caused by the Provider round; current timing evidence is insufficient.

### 2. ACPT-P22-CTX-004

**Fact:** the next turn does not carry a general assistant-offered action/target. `开始试听` is unresolved without such context; the Provider receives only the second user utterance. The existing `AudioSuspension` mechanism is specialized and does not implement general offered-action continuation.

**Failure layers:** Context / Authority / Selection.

**Root cause status:** `confirmed`.

No Core Acceptance behavior failure was confirmed in this run.

## L. Regression Candidates

No test or contract is changed in this run.

Because root cause is confirmed, the following are legitimate future regression candidates after the corresponding product behavior is intentionally fixed:

1. **WebShell deterministic playback-command parity**
   - `继续播放 / 暂停 / 下一首 / 上一首`
   - closed deterministic route;
   - Provider rounds = 0;
   - direct action;
   - readback where required.

2. **Target-bound conversational offered-action continuation**
   - assistant offer + target;
   - user short acceptance;
   - pending authority preserved;
   - same target executes;
   - no unnecessary clarification.

These are regression **candidates**, not newly added tests.

Scene/mood/cold-start remain evidence-coverage candidates rather than confirmed regressions.

## M. Current Version Scorecard

| Metric | Current Version | Previous Version | Delta | Evidence / qualification |
|---|---:|---:|---:|---|
| Core Acceptance Pass Rate | **46.15% (6/13)** | N/A | N/A | Strict `PASS` outcomes only. Remaining Core outcomes are 4 `PARTIAL-EVIDENCE` + 3 `UAT-REQUIRED`, **0 Core FAIL**. |
| P0 Acceptance Pass Rate | **50.00% (6/12)** | N/A | N/A | Strict PASS only; P1 Cold Start excluded. |
| Regression Pass Rate | **100.00% (5374/5374)** | N/A | N/A | Owner-verified real macOS engineering baseline; not rerun here. |
| Wrong Action Rate | **N/A** | N/A | N/A | No current real-action identity-chain denominator. No wrong target observed. |
| Fail-closed Correctness | **N/A** | N/A | N/A | Automated evidence passed, but current real Preview/playback UAT denominator is incomplete. |
| Unnecessary Clarification Rate | **N/A** | N/A | N/A | One confirmed Extension continuation failure; no corpus-wide denominator. |
| Tool Calling Success Rate | **N/A** | N/A | N/A | Existing test/journal contracts pass, but no unified current-run journal aggregation. |
| No-result Correctness | **100.00% (1/1 fixed Core no-candidate task)** | N/A | N/A | `ACPT-FAIL-001` passed all 4 mapped tests. |
| Owner UAT Pass Rate | **N/A** | N/A | N/A | No Owner UAT was executed as part of P22-EVAL-03; three Core tasks remain `UAT-REQUIRED`. |
| P50 Latency | **N/A** | N/A | N/A | No product-valid timing sample. |
| P95 Latency | **N/A** | N/A | N/A | No product-valid timing sample. |

### Interpretation

The low strict Core Acceptance Pass Rate is primarily an **evidence-completeness score**, not a failure rate:

```text
PASS             6
PARTIAL-EVIDENCE 4
UAT-REQUIRED     3
FAIL             0
```

The Product Acceptance run nevertheless produced new value by exposing two Extension behavior failures that a green `5374 / 5374` engineering regression does not express at the Product Task layer.

## N. Files Created

This run creates only:

```text
eval/runs/P22-EVAL-03-FIRST-RUN.md
```

No YAML snapshot was necessary for the first run because all fixed task IDs, outcomes, evidence classifications, run identity and scorecard are represented explicitly in this report.

Production changes: `0`.

Test changes: `0`.

Corpus contract changes: `0`.

Scorecard-template changes: `0`.

## O. Verification Commands / Evidence

### Corpus-mapped automated evidence

All 60 unique mapped tests passed, representing all 63 `existing_tests` references across the 15 tasks.

Execution was split into small existing-test batches only to fit the execution environment; no test logic was copied or rewritten.

### Extension route controlled runtime

A current-code standalone WebShell fixture was used only as evidence collection. No production/test file was modified.

Observed:

```text
继续播放 -> ProviderAgentLoop, no adapter action
暂停     -> deterministic pause, adapter pause
下一首   -> ProviderAgentLoop, no adapter action
上一首   -> ProviderAgentLoop, no adapter action
```

### Extension continuation controlled runtime

A recording Provider was used with the normal WebShell flow.

Offer `试听 Wendy` followed by `开始试听`:

```text
Provider call 1 messages: [user first-turn text]
Provider call 2 messages: [user "开始试听"]
adapter actions: []
```

Offer `继续播放` followed by `继续`:

```text
Provider call 1 messages: [user first-turn text]
Provider call 2 messages: [user "继续"]
adapter actions: []
```

The prior assistant offer is not carried as a general pending action.

## P. Run Status

```text
P22-EVAL-03 FIRST PRODUCT ACCEPTANCE RUN
COMPLETE

Core:
6 PASS
4 PARTIAL-EVIDENCE
3 UAT-REQUIRED
0 FAIL

Extensions:
0 PASS
0 PARTIAL-EVIDENCE
0 UAT-REQUIRED
2 FAIL

Production fixes:
NONE

Regression tests added:
NONE

Eval Runner:
NOT IMPLEMENTED

P22-S1.12:
NOT STARTED

P22-S2:
NOT STARTED
```
