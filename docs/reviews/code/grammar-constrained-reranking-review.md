# Code Review: grammar-constrained-reranking

## Verdict: APPROVED
## Reviewer Stance: Team — Proposer + Code Review Partner
## Date: 2026-03-13
## Mode: Agent Team (message-gated incremental review)

## PRP Compliance

| PRP Task | Commit | Status |
|----------|--------|--------|
| Task 1: RerankConfig and RerankResult types | `2d03d3f` | Implemented as specified |
| Task 2: generate_routing_grammar() | `b70fcb7` | Implemented as specified |
| Task 3: generate_routing_prompt() | `2bda6d4` | Implemented as specified |
| Task 4: ToolRouter class | `5a087df` | Implemented as specified |
| Task 5: SamplingSession integration | `8715917` | Implemented as specified |
| Task 6: Rerank telemetry fields | `40e0d16` | Implemented as specified |
| Task 7: __init__.py exports | `4e737eb` | Implemented as specified |

All 7 PRP tasks implemented in 7 atomic commits. Each commit contains tests + implementation together. TDD loop followed (RED -> GREEN -> REFACTOR -> COMMIT).

## Issues Found

### 1. Grammar text caching stores text, not mutable GrammarState
**Category:** Performance / Correctness
**Severity:** Nit (correctly handled)
**Location:** `src/tgirl/rerank.py:45`
**Details:** The plan review (YP3) noted that `GrammarState` is mutated by `advance()`, so the cache must produce fresh instances. The implementation correctly caches grammar TEXT in `_routing_grammar_cache: dict[tuple[str, ...], str]` and creates fresh `GrammarState` via `grammar_guide_factory()` on each `route()` call. This is the right design.
**Resolution:** No action needed — correctly implemented per plan review guidance.

### 2. Empty snapshot handled via delegation to generate_routing_grammar
**Category:** Logic
**Severity:** Minor
**Location:** `src/tgirl/rerank.py:93-95`
**Details:** When all tools are quota-exhausted, `route()` delegates to `generate_routing_grammar(filtered_snapshot)` which raises `ValueError`. This is indirect but correct — the error message comes from `generate_routing_grammar`, not `route()` itself.
**Resolution:** Acceptable — the ValueError is raised as specified in PRP Task 4.

### 3. Formatting-only changes in sample.py diff
**Category:** Convention
**Severity:** Nit
**Location:** `src/tgirl/sample.py` (multiple locations)
**Details:** The sample.py integration commit includes ruff formatting changes to existing code (line wrapping adjustments). These are not functional changes but add noise to the diff.
**Resolution:** Acceptable — consistent with running `ruff format` as part of validation gates.

## What's Done Well

- **Clean modular decomposition:** Each PRP task maps to exactly one commit, each commit touches exactly the specified files
- **Grammar text caching:** Correctly stores text rather than mutable GrammarState objects, addressing plan review YP3
- **Quota-aware routing:** Filters exhausted tools before grammar generation, satisfying PRD AC8
- **Inline snapshot filtering:** After routing, builds restricted snapshot from existing quota-adjusted data rather than re-snapshotting from registry (plan review YP5)
- **Empty hooks for routing pass:** Explicitly passes `hooks=[]` to prevent session hooks from misbehaving on the tiny routing grammar (plan review YP2)
- **Tokenizer encode injection:** `tokenizer_encode` added to `ToolRouter.__init__` for prompt tokenization (plan review YP4)
- **Test coverage:** 32 new tests across 5 test files, 476 total tests passing
- **`from __future__ import annotations`** in all new modules
- **Structlog logging** with structured fields in rerank module
- **Frozen Pydantic models** for all new types

## Summary

All 7 PRP tasks implemented faithfully with no spec deviations. All 5 plan review yield points addressed in the implementation. 476 tests pass. The feature is entirely additive — no existing API signatures changed, `rerank_config=None` preserves original behavior. **APPROVED for merge.**
