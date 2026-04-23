# Plan Review: inference-irq-controller

## Verdict: REQUESTS CHANGES
## Reviewer Stance: Team — Senior Training Partner + Proposer (degraded)
## Date: 2026-04-22
## Mode: Agent Team (concurrent review + revision) — **convergence failed**

## Exchange Summary

Team mode partially failed. The `training-partner` teammate spawned but went idle after every message without producing yield points or claiming its task, despite three escalating nudges from the team lead. This matches the pattern of Claude Code team bugs #24316 (agent definitions not loading) / #32368 (model inheritance). The `proposer` teammate, during its idle wait, independently read PRD/PRP/CLAUDE.md and verified every line-number reference in the PRP against the actual codebase, then produced a self-audit listing 8 issues before shutdown.

Consequence: no dialectic occurred. No yield points were "defended" by the proposer, no PRP revisions were made, nothing was counter-argued. The proposer's self-audit is documented below as **open items for human review**, with the team lead having independently verified the most load-bearing claims in the codebase.

The PRP at `docs/PRPs/inference-irq-controller.md` is **unchanged** from commit `524ff77`.

## Yield Points Found (from proposer self-audit, team-lead verified)

### 1. Streaming `/v1/chat/completions` path does not use `SamplingSession`
**Severity:** Structural
**Evidence:** `src/tgirl/serve.py:1616-1779` — `stream_gen()` runs its own inline per-token loop (`for _ in range(max_tok): logits = ctx.forward_fn(token_ids) ...`) that never instantiates or uses `SamplingSession`. PRP Task 4 assumes `session.irq.raise_interrupt(IrqSource.CANCEL)` on disconnect, but there is no `session` object on the SSE path. As written, Task 4's acceptance criterion (AC#2 — SSE cancel-on-disconnect) cannot be satisfied by the described approach.
**Proposer Response:** Pre-identified; no revision made (team did not converge).
**PRP Updated:** No
**Recommendation before `/execute-*`:** Task 4 must be rescoped. Either (a) add a prerequisite refactor that routes the SSE path through `SamplingSession`, or (b) acknowledge the inline loop needs its own parallel `IrqController` wiring and spec it explicitly (handler installation, dispatch point at the top of the `for` loop, how it shares the one `ctx` across requests). Option (a) is cleaner but larger; (b) preserves scope but admits dual plumbing.

### 2. Task 7 (autotuner migration) misses two `_run_autotune_after_turn` call sites
**Severity:** Structural
**Evidence:** PRP Task 7 lists `serve.py:1347, 1366` as the autotuner call sites to replace with a `TURN_COMPLETE` handler. `grep -n _run_autotune_after_turn src/tgirl/serve.py` shows additional call sites at `serve.py:1702` (stream_gen stop branch) and `serve.py:1766` (stream_gen length branch). Leaving these unmigrated means the streaming path continues calling the bespoke hook directly, violating AC#11 ("autotuner migration preserves behavior via handler only").
**Proposer Response:** Pre-identified; no revision made.
**PRP Updated:** No
**Recommendation:** Extend Task 7's file list and approach to include `serve.py:1702, 1766`. Note: this yield point is entangled with #1 — the streaming path can't hold a `session.irq` until that path uses `SamplingSession` or has its own controller.

### 3. AC#6 (handler grammar swap doesn't corrupt KV cache) has no explicit test
**Severity:** Structural
**Evidence:** PRD AC#6 requires a test that swaps `grammar_state` mid-generation and verifies KV cache integrity via token-sequence equality under a fixed seed. Reading Tasks 1–11, no task carries this test: Task 1 tests the controller in isolation (no cache), Task 2/3 test CANCEL/RESTART_TURN (no grammar swap), Task 6 migrates `TransitionPolicy` but its parity test captures whole-pipeline BFCL output, not swap-vs-no-swap cache integrity. The load-bearing safety claim in PRD §5 ("Grammar swap mid-advance undefined behavior") has no test attached.
**Proposer Response:** Pre-identified; no revision made.
**PRP Updated:** No
**Recommendation:** Add a dedicated task (or fold into Task 1/2) that implements AC#6 explicitly: fixture with a handler that swaps `grammar_state`, run with and without the swap under `seed=0`, assert cache hit rate and post-swap generated tokens match expectation. Also add the PRD §5 invariant assertion (no grammar swap during `GRAMMAR_TRIGGER`→`GRAMMAR_EXIT` window) — currently a comment, not enforced code.

### 4. `threading.Event` vs `threading.Lock` discrepancy between PRD and PRP
**Severity:** Moderate
**Evidence:** PRD §3 Dependencies states "CANCEL cross-thread signaling uses `threading.Event`." PRP Task 1 Approach states "`threading.Lock` for `raise_interrupt` (cross-thread: FastAPI event-loop → sampling thread)." The PRP choice is defensible (the hot-path reads a deque atomically under GIL; dispatch polls rather than block-waits), but the contradiction with the PRD should be reconciled in writing.
**Proposer Response:** Pre-identified; no revision made.
**PRP Updated:** No
**Recommendation:** Task 1 should either adopt the PRD's `threading.Event` + lock combination (event for wakeup-latency guarantees, lock for deque consistency) and document the rationale, or the PRD should be amended to match. If the PRP choice stands, explicitly state in Task 1: "`raise_interrupt` holds a `threading.Lock` briefly to append to `collections.deque`; the sampling thread polls `dispatch_pending()` between tokens. No `threading.Event` is used because the maximum wait (one forward pass) is already the design's latency bound per AC#1."

### 5. Per-token overhead micro-bench (Task 11) only measures empty queue
**Severity:** Moderate
**Evidence:** Task 11 measures 10,000 calls of `dispatch_pending()` with **no registered handlers**. Real traffic always has at least a `TURN_COMPLETE` handler installed (autotuner after Task 7). The 50µs budget (AC#10) is a floor-not-ceiling measurement; the real per-token cost in production will be higher and unmeasured.
**Proposer Response:** Pre-identified; no revision made.
**PRP Updated:** No
**Recommendation:** Add a second micro-bench in Task 11 with the realistic handler set installed (`CANCEL` + `GRAMMAR_TRIGGER` + `TOOL_COMPLETE` + `TURN_COMPLETE`), measuring the amortized per-token cost under the actual production registration. Keep the empty-queue bench as the lower bound; add an upper bound with a separate assertion (e.g., ≤ 200µs at p99 under realistic handlers).

### 6. Phase 1 "no KV cache rollback" contract is unenforced
**Severity:** Minor
**Evidence:** PRD §2 and §5 state handlers "MAY NOT roll back KV cache in Phase 1." PRP Task 1 repeats this as a comment. Nothing structurally prevents a handler from truncating `token_history`; safety rests on absence of a `cache.fork()` API plus handler discipline. Per CLAUDE.md "Safety by construction, not validation" and "Sandbox is defense-in-depth," this contract warrants an enforcement mechanism.
**Proposer Response:** Pre-identified; no revision made.
**PRP Updated:** No
**Recommendation:** Add an invariant check in `InterruptController.dispatch_pending`: record `len(session.token_history)` before each handler, assert post-handler length is monotonically non-decreasing. Violation → structlog + treat as handler exception (→ BREAK per AC handler-exception policy).

### 7. Uncertainty Log item on `TransitionPolicy` class location is stale
**Severity:** Minor
**Evidence:** PRP §6 Uncertainty Log item 1 says "TransitionPolicy class locations" are uncertain. `grep -n 'class.*TransitionPolicy' src/tgirl/state_machine.py` definitively resolves this: `TransitionPolicy:53`, `DelimiterTransitionPolicy:64`, `BudgetTransitionPolicy:135`, `ImmediateTransitionPolicy:180`, `ConfidenceTransitionPolicy:409`, `LatchedTransitionPolicy:497`, `CompositeTransitionPolicy:597`. The uncertainty should be removed and Task 6's file list should include `src/tgirl/state_machine.py` as a certainty, not a "verify first."
**Proposer Response:** Pre-identified; no revision made.
**PRP Updated:** No
**Recommendation:** Edit §6 to remove item 1; edit Task 6 file list to state `src/tgirl/state_machine.py` without the "verify class location first" hedge.

### 8. Task 5 (provider-comparison) critical-path entanglement
**Severity:** Minor
**Evidence:** Task 5 produces launch-evidence (provider-comparison benchmark). It requires OpenAI + Anthropic API keys + credits, is subject to third-party API flakiness, and has no code-integration dependency on Tasks 2–4. As written it sits between Task 4 and Task 6 in sequence, which implies downstream tasks wait on it.
**Proposer Response:** Pre-identified; no revision made.
**PRP Updated:** No
**Recommendation:** Explicitly mark Task 5 as off the merge-blocking critical path in the Rollback Plan section; it produces launch-writeup evidence that must exist before any public claim references cancel behavior, but nothing downstream code-wise depends on its output. Task 6 should be reorderable before Task 5.

### 9. Line-number imprecision: `sample.py:1024` is inside a break branch
**Severity:** Minor (documentation correctness)
**Evidence:** PRP Task 2 says "Freeform loop dispatch at `sample.py:1024` (after token append + telemetry update)." Reading the file, `sample.py:1024` is the `break` statement inside the `if decision.should_transition:` branch — it wouldn't execute on the continuing-iteration path. The actual token-append is at `sample.py:994-996`. The dispatch point should be placed either immediately after line 996 (after `total_tokens += 1`) or at the bottom of the iteration (after line 1024, outside the break branch).
**Proposer Response:** Pre-identified; no revision made.
**PRP Updated:** No
**Recommendation:** Update Task 2's approach to cite the correct line (994-996 for token append, bottom of loop iteration for dispatch). This is low-risk because the implementer will catch it, but PRP precision matters for reviewability.

## What Holds Well

- **Dependency graph and file list are comprehensive.** Most references resolved on first Read; only the streaming-path blind spot (#1) and the stale uncertainty (#7) cut through.
- **PRD grounding is tight.** Every PRP task traces to an AC; every AC ties back to a PRD problem statement. The thread from "aborted requests actually stop" → AC#1,2 → Task 4 is clean.
- **Rollback plan is real.** Per-task commit granularity and the Task 11 perf-budget merge gate are both standard-tier-appropriate.
- **TDD discipline is respected.** Every task has RED-first tests named, test files identified, and a test command.
- **BFCL bit-parity gate is the right shape.** Capture-before-refactor golden fixture + Task 11 regression test is the correct pattern for a behavior-preserving migration.
- **Uncertainty Log captures real risks.** Even with item 1 stale, items 2–7 are well-scoped and honest about what the PRP does not yet know.

## Summary

The PRP is **directionally correct** and the architecture it describes is sound. The structural issues are in specific task scope and test coverage, not in the overall design. The blocker for `/execute-team` is the **streaming-path blind spot (#1)** — Task 4 as written cannot be implemented without upstream refactoring that the PRP does not acknowledge. The **missing AC#6 test (#3)** is the second blocker: PRD §5 identifies grammar-swap mid-advance as undefined behavior, and the PRP must have a test that would catch corruption.

**Team-mode convergence failed** (training-partner produced no yield points). This review is therefore one-sided: every issue above is a proposer self-audit, team-lead-verified, never challenged by an adversarial training partner. A human reviewer should treat these as hypotheses to prioritize rather than adjudicated findings.

**Verdict: REQUESTS CHANGES.** Before `/execute-team`:
1. Rescope Task 4 to address the SSE streaming path's non-SamplingSession architecture (yield #1).
2. Extend Task 7 to cover `serve.py:1702, 1766` (yield #2).
3. Add an explicit AC#6 test for handler grammar swap + KV cache integrity (yield #3).
4. Reconcile `threading.Event` vs `threading.Lock` between PRD and PRP (yield #4).

Yields #5–9 are recommended but not strictly merge-blocking.

**Retry recommendation:** run `/review-plan docs/PRPs/inference-irq-controller.md` (solo, non-team fallback) to get genuine adversarial push-back on these findings before revising, since the team dialectic did not fire.
