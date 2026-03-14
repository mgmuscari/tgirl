# Plan Review: grammar-constrained-reranking

## Verdict: APPROVED
## Reviewer Stance: Team — Senior Training Partner + Proposer
## Date: 2026-03-13
## Mode: Agent Team (concurrent review + revision)

## Yield Points Found

### 1. Quota-exhausted tools leak into routing pass
**Severity:** Moderate
**Evidence:** `_snapshot_with_remaining_quotas()` (`sample.py:581-601`) sets quota=0 but keeps exhausted tools in `snapshot.tools`. `generate_routing_grammar()` enumerates all tools as grammar alternatives, so the router could select an exhausted tool. Violates PRD AC8.
**Proposer Response:** Accepted. Added quota filtering in `route()` before grammar generation. Three new test cases added (exhausted tools excluded, all-exhausted raises ValueError, single remaining short-circuits).
**PRP Updated:** Yes — Task 4 `route()` step 1, Task 4 tests.

### 2. ToolRouter duplicates session dependencies; hooks unspecified
**Severity:** Moderate
**Evidence:** `ToolRouter.__init__` takes `forward_fn`, `embeddings`, etc. already held by `SamplingSession`. More critically, `run_constrained_generation` requires `hooks` but the PRP never specified what hooks the routing pass uses. `GrammarTemperatureHook` would misbehave on a trivial 3-7 alternative routing grammar.
**Proposer Response:** Partially accepted. Ownership concern rejected (shared references via constructor injection is standard composition, not duplication). Hooks gap accepted — `route()` now explicitly passes `hooks=[]` with rationale. Test case added.
**PRP Updated:** Yes — Task 4 `route()` step 6, Task 4 tests.

### 3. Routing grammar compilation cost unaddressed
**Severity:** Structural (HIGH)
**Evidence:** PRP jumped from "generate grammar text" to "call `run_constrained_generation`" without the `grammar_guide_factory` compilation step. Outlines CFGGuide compilation is non-trivial and would dominate wall time for a 1-3 token routing generation. No caching mechanism was specified.
**Proposer Response:** Accepted. Added explicit compilation step and `_routing_grammar_cache: dict[tuple[str, ...], GrammarState]` keyed on sorted tool names. Cache invalidates naturally when quota exhaustion changes routable tool set. Two cache test cases added.
**PRP Updated:** Yes — Task 4 `__init__`, Task 4 `route()` steps 4-5, Task 4 tests.
**Implementation note:** `GrammarState` is mutated by `advance()`, so the cache must produce fresh instances per `route()` call (store grammar text, not mutable state objects).

### 4. Routing prompt has no injection mechanism
**Severity:** Moderate
**Evidence:** `generate_routing_prompt()` produces a string but `run_constrained_generation` only accepts `list[int]` token IDs. `ToolRouter.__init__` had `tokenizer_decode` but not `tokenizer_encode`. Without encoding, the routing prompt could never reach the model — Task 3 would be dead code.
**Proposer Response:** Accepted. Added `tokenizer_encode` to `ToolRouter.__init__`. Added explicit prompt tokenization step in `route()`. Task 5 forwards `tokenizer_encode` from `SamplingSession`. Test case added.
**PRP Updated:** Yes — Task 4 `__init__` signature, Task 4 `route()` step 3, Task 5 integration code.

### 5. Re-snapshot after routing loses quota-adjusted state
**Severity:** Minor (correctness bug)
**Evidence:** Task 5 called `self._registry.snapshot(restrict_to=...)` after routing — a fresh snapshot with original quotas, discarding cycle-adjusted values from `_snapshot_with_remaining_quotas()`. A tool with quota=3 called twice would appear as quota=3 instead of quota=1.
**Proposer Response:** Accepted. Replaced re-snapshot with inline filtering of the existing quota-adjusted snapshot. Test case verifying adjusted quotas survive restriction.
**PRP Updated:** Yes — Task 5 integration code, Task 5 tests.

## What Holds Well

- **Clean modular decomposition:** Tasks 1-3 are independent of sample.py and can proceed immediately
- **Honest uncertainty log** with correct risk assessment (branch dependency, top-K strategy, composition limitations)
- **Genuinely additive/reversible rollback plan** — no existing API signatures change
- **Determinism preserved** — same snapshot produces same routing grammar
- **Reuses existing infrastructure** — `GrammarState` protocol, `run_constrained_generation`, OT transport
- **Proposer was responsive and thorough** — 4/5 yield points fully accepted, 1 partially accepted, with concrete PRP revisions and 12 new test cases

## Summary

The PRP had a sound architectural vision but its integration layer (Tasks 4-5) had 5 gaps where pseudocode didn't match actual function signatures and data flow. All 5 yield points were addressed in PRP revisions:

- **Correctness bugs fixed:** quota leakage (YP1), quota loss on re-snapshot (YP5), dead-code routing prompt (YP4)
- **Performance risk mitigated:** grammar compilation caching (YP3)
- **Design gap closed:** explicit empty hooks for routing pass (YP2)

The revised PRP now has explicit data flow from routing prompt generation through tokenization, compilation, cached grammar state, and quota-aware filtering. **APPROVED for implementation.**
