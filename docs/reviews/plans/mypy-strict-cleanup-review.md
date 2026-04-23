# Plan Review: mypy-strict-cleanup

## Verdict: APPROVED
## Reviewer Stance: Team — Senior Training Partner + Proposer
## Date: 2026-04-22
## Mode: Agent Team (concurrent review + revision)

## Exchange Summary

10 yield points raised by the training partner in parallel at the start of the exchange. Proposer addressed each in sequence, revising the PRP where valid and pushing back with code-level evidence where the training partner's claim was overstated. Training partner conceded both pushback cases after re-verifying. Net: 6 accepted outright + 2 partial accepts + 2 self-retracted by training partner. Commit `3bbc953` on `feature/mypy-strict-cleanup` carries all revisions.

**Baseline verified during review:** `mypy src/ --ignore-missing-imports 2>&1 | grep -c "error:"` = 100 errors across 18 files (PRP originally said 17; corrected).

## Yield Points Found

### 1. `serve.py:88` misclassified as a tuple-unpack bug
**Severity:** Structural
**Evidence:** `_load_mlx_model` at `serve.py:84-89` unpacks a canonical 2-tuple from `mlx_lm.load`. No batch/seq_len/d_model anywhere in that path. mypy's `[misc]` error is a stub gap in `mlx_lm`, not a runtime bug. Original Task 12a would have prescribed a regression test for non-existent broken behavior.
**Proposer Response:** Accepted. Reclassified as stub gap.
**PRP Updated:** Yes. Task 12a removed; `serve.py:88` moved to Task 13 with `# type: ignore[misc]  # mlx_lm.load stub gap`. Task 12 sub-items renumbered. Error-drop target adjusted 6–8 → 4–6.

### 2. `lingo/types.py:128` misclassified as a str/set confusion bug
**Severity:** Structural
**Evidence:** Line 117 declares `anc = {name}` (set[str]); line 128's `for anc in ancs:` shadows the outer `anc` with a `str`. mypy is correctly flagging the name collision, but the runtime behavior is intentional and correct — the loop's intent is just confused by the shadowed name. A one-line rename (`anc` → `ancestor` in the loop) fixes it without any behavior change or regression test.
**Proposer Response:** Accepted. Reclassified as scope-shadowing cleanup.
**PRP Updated:** Yes. §2 "By rule" bullet updated; Task 10 heading changed to "Union narrowing & scope-shadowing cleanup"; Task 10 approach specifies the rename.

### 3. Task 6 reinvents `ForwardResult` instead of reusing the existing one
**Severity:** Structural
**Evidence:** `cache.py:21` already defines `ForwardResult(NamedTuple)` with `logits: Any, probe_alpha: Any | None = None`. `make_steerable_mlx_forward_fn` (`cache.py:309-408`) already returns it. Task 6 as originally written would have created a duplicate type, yielding import confusion between `cache.ForwardResult` and `sample_mlx.ForwardFnResult`.
**Proposer Response:** Accepted. Rewrote Task 6 to reuse the existing NamedTuple plus define a `SteerableForwardFn` protocol for the call-site contract, and unify the non-steerable `make_mlx_forward_fn` factory to also return `ForwardResult` (eliminating the `mx.array | ForwardResult` polymorphism that drives the 3 `attr-defined` errors at `sample_mlx.py:491-492`).
**PRP Updated:** Yes. Task 6 heading, approach, tests, and commit message all rewritten. Error-drop target 3 → 4 (the `steering=` call-arg at `sample_mlx.py:490` now resolves in this task).

### 4. Per-file error counts in PRP §2 don't match reality
**Severity:** Structural (claimed)
**Evidence offered:** `grep "^src/"` against mypy output showed `state_machine.py: 14`, `format.py: 14` etc. — different from PRP's `state_machine.py: 8, format.py: 6`.
**Proposer Response:** Rejected with evidence. The training partner's grep filter inadvertently counted `note:` continuation lines on top of error lines. Correct grep (`grep "error:"`) returns 100 errors matching PRP's table exactly. **Training partner retracted after re-running.**
**PRP Updated:** Minor — §1 added a methodology note specifying `grep "error:"` (not `grep "^src/"`) for reproducibility.

### 5. `ConfidenceMonitorProto` is missing — 5 `attr-defined` errors unaccounted for
**Severity:** Structural
**Evidence:** `confidence_monitor: object | None = None` at `sample_mlx.py:429` drives 5 `attr-defined` errors at `sample_mlx.py:535, 640, 642, 643, 676` on methods `should_checkpoint`, `record_log_prob`, `should_backtrack`, `backtracks_remaining`, `record_backtrack`. Concrete implementation lives at `state_machine.py:367-388`. Zero PRP/PRD mentions — the Task 5 RED target of 8 undershoots by 5.
**Proposer Response:** Accepted. Folded into Task 5 rather than spinning a separate Task 5b because both `controller` and `confidence_monitor` are `object | None` parameters on the same function signature. Full 5-method Protocol surface specified; Protocol co-located with concrete class at `state_machine.py`.
**PRP Updated:** Yes. Task 5 heading "+ ConfidenceMonitorProto"; files expanded (added `state_machine.py`); Tests RED: 8 → 13; added `test_monitor_satisfies_proto`; commit message updated; new Uncertainty Log bullet on protocol-surface risk.

### 6. `# type: ignore` budget likely overruns the cap of 10
**Severity:** Moderate
**Evidence:** Naive projection — Protocol additions, union narrowings, and stub-gap mitigations all trended toward `# type: ignore` proliferation.
**Proposer Response:** Partial accept. Re-tallied: the projection incorrectly assumed Protocol additions would consume budget (they don't — Protocols resolve via typing). Actual budget usage: 8 of 10 (Task 7: 2 for llguidance.mlx; Task 10: 4 for MLX `stream=` kwarg stub gaps; Task 13: 1 for `RestrictedPython` dynamic base + 1 for `mlx_lm.load` stub gap). Task 5, Task 6, Task 12c all resolve via typing — zero ignores.
**PRP Updated:** Yes. Uncertainty Log rewritten with 8/10 breakdown + escalation path (use `[[tool.mypy.overrides]]` if >10). Adopted training partner's proposed commit-footer convention `mypy: X → Y (delta)` — now required in §4.

### 7. Task DAG hidden
**Severity:** Moderate
**Evidence:** 4 claimed must-precede relations between tasks (Task 6 → Task 13 on `steering=`; Task 7 → Task 9 on tokenizer proto; Task 12c consolidation; Task 5 → Task 12a on `controller` narrowing).
**Proposer Response:** Partial accept. DAG request accepted. 3 of 4 specific claims accepted as hard dependencies; the Task 5 → Task 12a claim downgraded to "preferred, not hard" because mypy's flow narrowing doesn't span two separate `if controller is not None` blocks regardless of controller type — the real fix is `estradiol_alphas` narrowing from None, independent of the controller Protocol. Training partner conceded.
**PRP Updated:** Yes. New subsection "Task dependency DAG (must-precede relations)" in §3, inserted between the section header and Task 1, with 4 relations and recommended execution order.

### 8. `rerank.py:143` violates CLAUDE.md's "no cross-framework conversions" rule
**Severity:** Moderate
**Evidence:** `rerank.py:125-151` passes a torch `GrammarState` into `run_constrained_generation_mlx`; the fallback at `sample_mlx.py:506-507` papers this over with `valid_mask = mx.array(valid_mask_torch.numpy())` — exactly the torch→numpy→mx conversion CLAUDE.md forbids (2026-03-14 gotcha). Original Task 10 would have `cast()` around the mismatch, preserving the violation.
**Proposer Response:** Accepted. `serve.py:48` already exposes `mlx_grammar_guide_factory` as a separate field, so the correct fix routes MLX-path rerank through the mlx factory rather than coercing the torch result. Promoted to Task 12c.
**PRP Updated:** Yes. `rerank.py:143` removed from Task 10; Task 12 gained a new 12c with two disambiguating possibilities (backend-aware factory vs always-torch) resolved by one regression test (`test_mlx_backend_uses_mlx_grammar_state`). **Fallback at `sample_mlx.py:506-507` deleted regardless** — cross-framework conversion is not a legitimate runtime path. Commit message updated; Uncertainty Log documents possibilities + 30-LoC budget for end-to-end factory wiring.

### 9. Pre-commit gap mitigation is "trust the operator"
**Severity:** Moderate
**Evidence:** This branch is based on `main` (pre-framework-sync), so the OLD pre-commit hook is active — mypy does NOT run on commit locally. PRP original text acknowledged this in Uncertainty Log but only committed to "interlocutor enforces discipline" — a hope, not a gate.
**Proposer Response:** Accepted.
**PRP Updated:** Yes. New §4 subsection "Per-task mypy gate (team-mode)": interlocutor (training partner in `/execute-team`) runs `mypy src/ 2>&1 | grep -c "error:"` between every proposer commit and blocks the next task if the count rises. Gate lives in team protocol, not a git hook. Combined with commit footer `mypy: X → Y (delta)` from yield #6 for audit traceability.

### 10. Minor cleanups (bundled)
**Severity:** Minor
**Evidence:**
- **10.1:** PRP references `tests/test_calibrate.py` as "if exists; else add one" — file already exists.
- **10.2:** `serve.py:89` should be `:88`.
- **10.3:** Task 2 and Task 5 both cite `sample_mlx.py:430` / `:434` — claimed conflation.
- **10.4:** `sample.py:1090` acknowledged, no action requested.
- **10.5:** File count "17" should be "18".

**Proposer Response:**
- 10.1: Accepted. Uncertainty Log bullet deleted.
- 10.2: Already fixed in yield #1 round.
- 10.3: Rejected with evidence. Task 2 correctly cites `:430` (grammar_guide_factory `[type-arg]`); Task 5 correctly cites `:434` (controller declaration). Different params, different lines, different tasks. **Training partner retracted.**
- 10.4: Acknowledged.
- 10.5: Accepted. §1 fixed.
**PRP Updated:** Yes, where accepted.

## What Holds Well

- **Codebase anchors are tight.** Every file path and line number in PRP §2 and §3 was verified during review and matched the codebase. The only anchor that drifted was `serve.py:89 → :88`, fixed in yield #1.
- **Per-task TDD framing is correct.** RED = mypy error count for the subset; GREEN = count → 0; COMMIT = test + impl together. Right discipline for annotation-heavy work.
- **Rollback Plan is concrete and per-task.** Each task is a single commit; revert-range granularity is clear.
- **Uncertainty Log is substantive, not theater.** Lists real unknowns (MLX stub completeness, pydantic strict-init cascade) with escalation paths.
- **Protocol co-location pattern.** PRP §6 OQ#3 proposes Protocols live next to concrete implementations (`EstradiolControllerProto` in `estradiol.py`, `ConfidenceMonitorProto` in `state_machine.py`). Consistent with existing `GrammarState` (sample.py) and `TransitionPolicy` (state_machine.py) patterns.
- **Baseline snapshot task** (Task 1) gives every subsequent task a diff-able reference for delta tracking — pairs well with the commit footer convention adopted in yield #6.
- **Latent-bug discipline.** PRP originally mandated regression tests for Task 12 bugs; review pressure improved this further by reclassifying two false positives out of Task 12 (yields #1, #2) and promoting one real cross-framework violation in (yield #8).

## Process Notes

- **Two self-retracted yields (#4, #10.3).** Training partner raised both without re-verifying evidence; proposer pushed back with file:line citations; training partner re-ran and conceded. Healthy dialectic signal — both sides checking each other's work.
- **PRD hygiene.** PRD §1 line 16 carried the same `serve.py` / `lingo/types.py` misclassification as the PRP draft. Amended in commit `3bbc953` with a one-line note pointing at the PRP for authoritative classification.

## Summary

All 10 yield points resolved. Two structural corrections tightened the Task 12 scope (`serve.py:88` reclassified as stub gap, `lingo/types.py:128` reclassified as scope-shadowing). One uncovered a missing Protocol cluster (`ConfidenceMonitorProto` — yield #5) that would have left 5 errors unaddressed at execute time. One caught a duplicate type definition (yield #3, existing `cache.ForwardResult`). One surfaced a CLAUDE.md invariant violation that the original PRP would have papered over (yield #8, cross-framework conversion in `rerank.py:143`). The moderate accepts (yields #6, #7, #9) added enforcement gates — explicit task DAG, per-task mypy gate for team mode, commit-footer audit trail — that make the execute phase verifiable.

**Verdict: APPROVED.** Recommend `/execute-team docs/PRPs/mypy-strict-cleanup.md`.
