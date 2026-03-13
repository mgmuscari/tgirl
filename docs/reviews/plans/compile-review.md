# Plan Review: compile

## Verdict: APPROVED
## Reviewer Stance: Team — Senior Training Partner + Proposer
## Date: 2026-03-12
## Mode: Agent Team (concurrent review + revision)

## Yield Points Found

### 1. `catch` vs `except` syntax inconsistency
**Severity:** Medium
**Evidence:** TGIRL.md:308 uses `catch` for error handling, but Hy 1.x uses `except`. PRP listed `catch` in the allowed-keyword list (Task 3) but tested with `except` in Task 5 expressions. Grammar emits `catch` per spec, but Hy parser would reject it.
**Proposer Response:** Accepted. Added `_normalize_hy_source()` in Task 2 to translate `catch` to `except` before Hy parsing. Updated keyword lists in Task 3 to accept both forms with explanatory note. Changed Task 5 try/catch test to use spec syntax, exercising the normalization path.
**PRP Updated:** Yes — Tasks 2, 3, 5

### 2. Result capture from `exec()` unresolved
**Severity:** High (structural)
**Evidence:** TGIRL.md section 5.5 requires a result accumulator. PRP Uncertainty Log item 3 flagged it as "critical implementation detail" but no task resolved it. Would block Task 8 and require backtracking to Tasks 4 and 6.
**Proposer Response:** Accepted. Added `_tgirl_result_` sentinel to sandbox (Task 6), `_inject_result_capture()` AST rewriter in Task 8 that rewrites last `ast.Expr` to assignment. Ordered injection AFTER security analysis so it doesn't pollute the trust boundary. Marked Uncertainty Log item 3 as RESOLVED.
**PRP Updated:** Yes — Tasks 1, 4, 6, 8; Uncertainty Log item 3

### 3. Blanket attribute access rejection contradicts spec
**Severity:** High (structural)
**Evidence:** TGIRL.md line 388 allows safe attribute access patterns on dict-like results; section 5.4 only rejects dunder attributes. PRP Task 3 blanket-rejected all attribute access, breaking the grammar-compile contract.
**Proposer Response:** Accepted. Split into dunder rejection (blocked) vs non-dunder access (allowed). Updated Task 3 with safety ownership annotation — grammar is primary guard (constrains to type-derived field names), static analyzer is defense-in-depth. Updated Task 4 RestrictedPython subclass guidance to override `visit_Attribute` for non-dunder access. Added tests for both cases.
**Follow-up (Round 2):** Training partner pressed on non-dunder escape vectors (`.gi_frame`, `.cr_frame`, `.tb_frame` on generators/coroutines reaching `f_globals`). Proposer documented why these are unreachable (grammar constrains to type-derived names, tool returns are Pydantic models/primitives), added forward-looking note that loosening grammar attribute productions would require upgrading from blocklist to allowlist, and flagged for security audit verification.
**PRP Updated:** Yes — Tasks 3, 4

### 4. `pmap` error/failure semantics unspecified
**Severity:** Medium
**Evidence:** Task 5 only tested pmap happy path. No specification for what happens when one tool in a pmap list raises, no interaction with per-tool timeouts addressed.
**Proposer Response:** Partially accepted. Added fail-fast semantics (stop and re-raise on first error), documented that timeout wrappers are applied at sandbox construction before pmap sees callables, added three error-path tests (failing tool raises, try/catch recovery, timeout-wrapped tools). Symbol resolution concern was rejected as a non-issue (training partner concurred).
**PRP Updated:** Yes — Task 5

### 5. `insufficient-resources` conflated with `PipelineError`
**Severity:** Medium
**Evidence:** TGIRL.md lines 287, 323 treat `insufficient-resources` as a valid grammar alternative (alongside `pipeline` and `single_call`), not an error. PRP Task 5 returned it as `PipelineError`, making model-chose-not-to-act indistinguishable from actual failures.
**Proposer Response:** Accepted. Added `InsufficientResources` as a distinct frozen Pydantic model. Updated `run_pipeline` return type to three-way union: `PipelineResult | InsufficientResources | PipelineError`. Carried through to Tasks 1, 5, 8, 9 (exports).
**PRP Updated:** Yes — Tasks 1, 5, 8, 9

## What Holds Well

- Three-layer defense-in-depth architecture is well-structured and clearly separated
- Task ordering follows a logical dependency chain (types -> parse -> analysis -> sandbox -> timeout -> assembly -> integration)
- Uncertainty Log is honest about known unknowns — items 3 and 8 were genuine risks
- Hy API research section is thorough and accurate against Hy 1.x
- Integration with existing codebase (PipelineError at types.py:207-216, ToolRegistry API) is well-referenced with correct line numbers verified against actual code
- The decision to use RestrictingNodeTransformer directly on hy.compile() output rather than source round-tripping is sound
- Test coverage is comprehensive with clear test class organization per feature

## Summary

All 5 yield points were addressed — 4 fully accepted, 1 partially accepted. The two HIGH-severity issues (result capture and attribute access) were genuine structural weaknesses that would have caused implementation backtracking. Both were resolved with changes that strengthen the plan without adding unnecessary complexity. One follow-up round was used on the attribute access concern to probe non-dunder escape vectors, resulting in documented safety invariants and a security audit flag. The PRP is ready for execution.
