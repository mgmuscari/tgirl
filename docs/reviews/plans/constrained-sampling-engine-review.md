# Plan Review: constrained-sampling-engine

## Verdict: APPROVED
## Reviewer Stance: Team — Senior Training Partner + Proposer
## Date: 2026-03-12
## Mode: Agent Team (concurrent review + revision)

## Yield Points Found

### 1. Logit processing order contradicts the spec
**Severity:** HIGH (Structural)
**Evidence:** TGIRL.md section 8.5 specifies OT redistribution (step d) before temperature/top-p (step e) and before penalties (step f). The PRP's monolithic `apply_sampling_params` bundled all post-logit operations into one function with no ordering relationship to OT.
**Proposer Response:** ACCEPTED. Split into `apply_penalties()` (pre-OT: repetition, presence, frequency penalties, logit bias) and `apply_shaping()` (post-OT: temperature, top-k, top-p). Added explicit 9-step per-token processing order in Task 5. Documented penalty ordering deviation from spec with empirical validation needed.
**PRP Updated:** Yes — Tasks 4, 5, 8, Uncertainty Log item 6 added.

### 2. DelimiterDetector buffer growth is unbounded
**Severity:** MEDIUM
**Evidence:** PRP Task 6 `_buffer` list grew with every token but was never pruned. `len(self.delimiter) * 4` heuristic mixed character and token counts.
**Proposer Response:** ACCEPTED. Replaced token buffer with decoded-text sliding window pruned to `2 * len(delimiter)` characters. Added resource test for bounded memory after 10,000 tokens.
**PRP Updated:** Yes — Task 6 approach and tests revised.

### 3. Top-p scatter implementation is incorrect
**Severity:** HIGH (Structural)
**Evidence:** PRP Task 4 `sorted_logits.scatter(0, sorted_indices, sorted_logits)` is self-referential — scatters from and into the same tensor. Produces garbled output. Uncertainty log misidentified as "numerical edge cases."
**Proposer Response:** ACCEPTED. Fixed to `result = torch.empty_like(sorted_logits); result.scatter_(0, sorted_indices, sorted_logits)`. Strengthened test to verify positional correctness. Updated uncertainty log.
**PRP Updated:** Yes — Task 4 approach, tests, and Uncertainty Log item 3 revised.

### 4. No mechanism for cross-cycle quota tracking
**Severity:** MEDIUM
**Evidence:** TGIRL.md section 3.3 requires quota persistence across cycles. `RegistrySnapshot` is frozen. `PipelineResult` doesn't report invocation counts. PRP said "update quota state" without specifying how.
**Proposer Response:** ACCEPTED. Added mutable `_consumed_quotas` state, `_count_tool_invocations()` Hy AST walk, `_snapshot_with_remaining_quotas()` method. Updated `SamplingResult` and `ToolCallRecord` with quota fields. Three specific tests added.
**PRP Updated:** Yes — Task 7 approach and tests significantly expanded.

### 5. Greedy tie-breaking is undocumented
**Severity:** LOW (Minor)
**Evidence:** `torch.argmax()` picks first index on ties with no documentation of this policy.
**Proposer Response:** PARTIALLY ACCEPTED. Rejected "bug" framing — standard greedy behavior, post-OT ties are measure-zero. Accepted documentation and test improvement. Added code comment and test for tied-maxima behavior.
**PRP Updated:** Yes — Task 4 tests expanded.

### 6. Branch name mismatch
**Severity:** LOW (Minor)
**Evidence:** PRP line 31 claimed branch already existed; actual state differed.
**Proposer Response:** ACCEPTED. Changed to "target branch, to be created from main at execution time."
**PRP Updated:** Yes — Line 31 revised.

## What Holds Well

- **Protocol-based abstractions** for `GrammarState` and `InferenceHook` provide clean Outlines decoupling — the `grammar_guide_factory` dependency injection is well-designed
- **Task sequencing** is logical: types (T1) → building blocks (T2-T4) → constrained core (T5) → delimiter detection (T6) → orchestrator (T7) → integration (T8)
- **Uncertainty log** is honest and identifies real risks (Outlines API stability, delimiter tokenization, embedding source)
- **Rollback plan** is clean — `sample.py` is a new file, `types.py` changes are additive, no existing module behavior modified
- **Proposer was responsive and thorough** in all revisions, including honest pushback on YP5

## Summary

All 6 yield points were addressed in the PRP revision. Both HIGH-severity issues (logit processing order and top-p scatter bug) were structurally fixed with strengthened tests. The MEDIUM-severity quota tracking gap now has a complete mechanism (mutable session state + Hy AST walk + reduced-quota snapshots). No unresolved concerns remain.

One noted deviation: penalty ordering (pre-OT rather than post-OT as in spec section 8.5) is documented as requiring empirical validation during benchmarking — this is logged in the Uncertainty Log and is non-blocking.

**APPROVED for execution.**
