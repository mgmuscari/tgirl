# Plan Review: inference-irq-controller

## Verdict: APPROVED
## Reviewer Stance: Team — Senior Training Partner + Proposer
## Date: 2026-04-22
## Mode: Agent Team (concurrent review + revision)

## Exchange Summary

Two revision rounds, genuine dialectic throughout. The training-partner raised 6 yield points in the first round, then pushed back on the proposer's initial 4a/4b scope compromise and added Y7 mid-round; the proposer's first-pass commit (`dfbf4a8`) dropped Y7 on the floor, and both the training-partner and team-lead caught the gap independently. A second revision (`23fdb51`) landed Y7 plus a mandatory 4a-i/4a-ii split plus three remaining clarifications. Task count grew 11 → 13; cumulative PRP diff +262 / -71.

**Earlier note on process:** the first team-mode attempt this session failed because every `.claude/agents/*.md` file restricted teammates from `SendMessage` and `TaskUpdate` in their `tools:` frontmatter (see CLAUDE.md Known Gotchas 2026-04-22). Fixed in commits `9736c82` and `8778e9f` before this successful run.

## Yield Points Found

### 1. SSE + non-streaming `/v1/chat/completions` path bypasses `SamplingSession`
**Severity:** Structural
**Evidence:** `src/tgirl/serve.py:1619-1779` (`stream_gen`) and `src/tgirl/serve.py:1253-1366` (`_generate_tokens`) each run their own inline MLX loops calling `ctx.forward_fn` directly. Neither instantiates `SamplingSession`. PRP Task 4 as originally written wired CANCEL via `session.irq.raise_interrupt(...)` — no session, no attach point, no way to satisfy AC#1/#2/#3 (the v0.2 launch differentiator).
**Proposer Response:** Accepted as structurally material. Split Task 4 into **4a** (route streaming + non-streaming chat completions through `SamplingSession`) and **4b** (the original cancel-on-disconnect, now well-defined). Training-partner then escalated: `SamplingSession` today has no streaming iterator API, and its `.run()` / `.run_chat()` run the full dual-mode/constrained/tool/grammar/transport/rerank machinery whereas the SSE path does plain steered freeform. Proposer split further into **4a-i** (add `SamplingSession.iter_tokens(prompt_tokens, *, enable_tool_calling=True)` with pre-capture golden fixture `plain_freeform_fixed_seed.json`) and **4a-ii** (route the chat completion endpoints through it with an OpenAI-boundary fixture). AC#1 honestly relaxed to "≤ 1 forward pass common case; ≤ 2 under adversarial cross-thread contention."
**PRP Updated:** Yes (both revisions: `dfbf4a8` then `23fdb51`).

### 2. Task 7 (autotuner migration) misses 2 of 4 call sites; none have a session in scope
**Severity:** Structural
**Evidence:** `_run_autotune_after_turn` is a closure defined inside `create_app` at `serve.py:1158`. Original PRP listed only `1347, 1366` (the latter off-by-one — actual site is `1365`) and missed `1702, 1766` inside `stream_gen`. None of the call sites run inside a `SamplingSession`. The PRP also ignored the write-before-next-read ordering invariant on `_autotune_state["next_*"]` — read by the *next* request's `_resolve_alpha` / `_resolve_temperature`.
**Proposer Response:** Accepted. Task 7 rewritten to cover all four sites with correct line numbers, uses `functools.partial` to capture `create_app` closure state (`_autotune_state`, `_steering_stats`, `_steering_config`, `_probe_cache`) explicitly when installing the `TURN_COMPLETE` handler. New regression test `test_write_before_next_read_invariant` and a 5–10 turn autotune trajectory fixture `tests/regression/autotune_fixtures/multi_turn_seed0.jsonl` + per-turn `next_*` state snapshots. Depends on Task 4a-ii to give every site a session.
**PRP Updated:** Yes (both revisions).

### 3. `controller=` kwarg collision with ESTRADIOL
**Severity:** Structural
**Evidence:** `run_constrained_generation` (`src/tgirl/sample.py:485, controller=:499`) and `run_constrained_generation_mlx` (`src/tgirl/sample_mlx.py:420, controller=:434`) already bind `controller` as the ESTRADIOL steering controller parameter (see `sample_mlx.py:467-495`). Original Task 2/3 dispatch points are inside these module-level functions with no `self._irq` in scope, and a HandlerFn contract of `(session, payload)` has no session to hand to handlers called from inside constrained loops.
**Proposer Response:** Accepted. `InterruptController.__init__(session)` now binds the session at construction; handlers still receive `(session, payload)` but the session comes from `self._session`, not a dispatch argument. Module-level gen functions gain a new `irq: InterruptController | None = None` kwarg. Naming disambiguated: `self._irq` (IRQ) vs `self._controller` (ESTRADIOL).
**PRP Updated:** Yes (`dfbf4a8`). Lifecycle note (strong-ref coupling documented, weakref deferred to Phase 2) landed in `23fdb51`.

### 4. Fast-path + TIMER source form a silent-bug shape
**Severity:** Moderate
**Evidence:** Task 1's `if not self._pending: return CONTINUE` fast-path combined with Task 9's TIMER source means scheduled timers never fire when the pending queue is empty. PRP did not include a test for "empty queue + past scheduled interval → timer fires."
**Proposer Response:** Accepted. Fast-path conditioned on a cached `_next_timer_fire: float | None` with `time.monotonic()` comparison. Task 9 now populates the scheduling surface that Task 1 consumes. New test `test_timer_fires_on_empty_queue_after_interval` covers the exact silent-bug case. Cross-thread `schedule()` semantics documented in second pass (writes under lock, fast-path read lockless via GIL-atomic single-slot read — stale read is a one-dispatch delay max, not a correctness bug).
**PRP Updated:** Yes (both revisions).

### 5. Drain-side thread safety was implicit (only `raise_interrupt` was locked)
**Severity:** Moderate
**Evidence:** Original Task 1 only lock-protected `raise_interrupt`. Three unsynchronized paths: (a) drain-in-flight + concurrent raise could arrive one forward pass late; (b) `mask()` mutation during an in-flight dispatch could produce mixed semantics; (c) Task 10's event-buffer cross-thread read could tear on larger-than-word payloads. AC#1/#4/#12 were probabilistic under contention.
**Proposer Response:** Accepted on all three sub-points. Snapshot-under-lock / invoke-handlers-outside-lock pattern spelled out in Task 1. Mask set snapshotted at dispatch start — changes mid-dispatch take effect on the next dispatch. Task 10 writes the event buffer under lock; reads take a snapshot copy under lock. AC#1 honestly relaxed to "≤ 1 forward pass common case; ≤ 2 under adversarial cross-thread contention." New Hypothesis stateful concurrent test added.
**PRP Updated:** Yes (`dfbf4a8`).

### 6. Per-token overhead micro-bench (AC#10) measures the wrong thing
**Severity:** Moderate
**Evidence:** Task 11 measured 10,000 empty-queue dispatches with zero registered handlers. Real production state after Tasks 6/7/8 migrate always has 6+ default handlers registered (`CANCEL`, `GRAMMAR_TRIGGER`, `GRAMMAR_EXIT`, `BUDGET_EXHAUSTED`, `TOOL_COMPLETE`, `TURN_COMPLETE`) — many with a non-trivial handler pending at least once per turn. The PRP's own Uncertainty Log flagged this and the test never operationalized it.
**Proposer Response:** Accepted. Task 11 replaced with four bench variants: `bench_empty_no_handlers` (50µs budget), `bench_empty_with_handlers` (50µs), `bench_one_raise_per_100_tokens` (mean over 100 loaded ≤ 500µs AND p95 over all 10,000 ≤ 100µs), plus observed `bench_turn_complete_handler_per_invocation` (warn > 1ms, no hard fail). Per-bucket histograms in the committed JSON result file.
**PRP Updated:** Yes (both revisions — bench statistic granularity clarified in `23fdb51`).

### 7. Task 8's `/generate` MODE_SWITCH_REQUEST migration is semantically mismatched
**Severity:** Structural
**Evidence:** `/generate`'s per-request overrides at `serve.py:835-897` are construction-time — they build a new `ToolRegistry` via `_filter_registry`, apply `model_copy` to transport/session config, instantiate a fresh `GrammarTemperatureHook`, and pass those already-baked values into `_run_session_chat` which constructs `SamplingSession` at `serve.py:599-613`. Raising `MODE_SWITCH_REQUEST` *before* `session.run()` on a session that already has the overrides baked in is either a no-op or requires mutating frozen state — including `ToolRouter`'s cached registry state at `sample.py:816-826`. Construction-time configuration and mid-session mutation are different substrates.
**Proposer Response:** Accepted. Task 8 rescoped to **TOOL_COMPLETE only** with an explicit "MODE_SWITCH_REQUEST rescoped" section (PRP:404) explaining the construction-time-vs-mid-session distinction. `MODE_SWITCH_REQUEST` reserved for genuine mid-session orchestrators in Phase 1 (no call sites yet; useful scaffold for future external orchestration). PRD revision flagged as a follow-up to either narrow the source's scope or add a concrete mid-session use case.
**PRP Updated:** Yes (`23fdb51`).

## What Holds Well

- **Dependency graph + codebase anchors are now tight.** All 13 tasks trace to PRD acceptance criteria, and every cited line number was verified against the codebase during review.
- **BFCL bit-parity capture-first pattern (Task 6)** is the right shape for behavior-preserving migrations.
- **Per-task commit atomicity + per-task rollback granularity** is preserved.
- **TDD discipline holds throughout.** Every task has RED-first tests named, test files identified, test command, and a concrete validation gate.
- **Honest acceptance criteria.** AC#1 was relaxed from "≤ 1 forward pass" to "≤ 1 common / ≤ 2 under contention" rather than accept a flaky test — the launch narrative still holds because both bounds are dramatically better than the provider-comparison baseline.
- **Non-trivial clarification bandwidth.** Session↔controller lifecycle, `schedule()` thread model, bench statistic granularity — all three handled in-line rather than punted.
- **Scope discipline under pressure.** Training-partner pushed for a hard 4a split; proposer initially resisted with a contingency note, then accepted when the audit surfaced the missing streaming API. No "fix later" shims — the 4a-i / 4a-ii split is load-bearing substrate work, not a compromise.

## Minor Polish Items (not blocking)

- **`docs/PRPs/inference-irq-controller.md:26`** still lists "`serve.py:835–897` — `/generate` per-request overrides → raise `MODE_SWITCH_REQUEST` (Task 8)" in the Codebase Analysis section, which is stale after Y7 rescoping. Worth deleting or annotating as "not migrated (Y7 — construction-time)."
- **4a-i ctx.bottleneck_hook coupling** logged in Uncertainty Log — will be resolved at implementation time.
- **PRD follow-up** recommended for three post-review items: AC#1 bound (Y5), AC#10 scope (Y6), MODE_SWITCH_REQUEST semantics (Y7). None block Phase 1 execution.
- **Handler timeout / deadline** — Phase 1 accepts this risk; flagged for `/review-code` attention.

## Summary

All seven yield points were addressed in the PRP revisions (`dfbf4a8` + `23fdb51`). Two structural concerns required a second revision pass because Y7 was dropped on the floor during the first commit — the team-lead and training-partner both caught the gap, and the proposer responded promptly. Net result: a PRP that is demonstrably more buildable than it was at review start — the SSE streaming path is explicitly routed through `SamplingSession` before CANCEL wiring, all migration targets are semantically well-posed, thread-safety and timer interactions are spelled out, and the perf budget tests the production shape of the system rather than an empty-queue idealization.

**Verdict: APPROVED.** Recommend `/execute-team` on the revised PRP.
