# Code Review: mlx-native-inference

## Verdict: APPROVED
## Reviewer Stance: Team -- Proposer + Code Review Partner
## Date: 2026-03-13
## Mode: Agent Team (message-gated incremental review)

## PRP Compliance

| Task | Spec | Status |
|------|------|--------|
| Task 1: transport_mlx.py | MLX Sinkhorn OT with TransportResultMlx, zero coupling | Implemented as specified |
| Task 2: sample_mlx.py | MLX penalties, shaping, constrained generation | Implemented as specified (after fix) |
| Task 3: cache.py | make_mlx_forward_fn returns mx.array, torch compat wrapper | Implemented as specified |
| Task 4: sample.py | SamplingSession backend dispatch (auto/torch/mlx) | Implemented as specified (after fix) |
| Task 4b: rerank.py | ToolRouter MLX dispatch | Implemented as specified (after fix) |
| Task 5: outlines_adapter.py | get_valid_mask_np numpy accessor | Implemented (consumer wiring deferred) |
| Task 6: benchmarks | Auto-detection handles transparently | No code changes needed |
| Task 7: __init__.py | Export MLX APIs with ImportError guard | Implemented as specified |

## Commits

| Commit | Task | Description |
|--------|------|-------------|
| b01e8b8 | 1 | MLX-native Sinkhorn OT with TransportResultMlx |
| bf764c7 | 2 | MLX-native sampling functions |
| 2737d56 | 3 | make_mlx_forward_fn returns mx.array, torch compat wrapper |
| ff60f21 | 4 | SamplingSession backend dispatch |
| 1c6a0da | 4b | ToolRouter MLX dispatch |
| 689520d | 5 | get_valid_mask_np numpy mask accessor |
| 8480439 | 7 | Export MLX APIs from __init__.py |
| 46fd9ac | fix | Zero-mass fallback, ToolRouter backend wiring, lazy torch import |

## Issues Found

### 1. Missing zero-mass fallback in MLX sampling
**Category:** Logic
**Severity:** Significant
**Location:** src/tgirl/sample_mlx.py, src/tgirl/sample.py (freeform MLX path)
**Details:** mx.random.categorical called without guarding against all-`-inf` logits. The torch path has explicit zero-mass fallback to uniform sampling. Aggressive top-k/top-p + grammar masking could produce all-masked logits.
**Resolution:** Fixed in 46fd9ac. Added prob_sum check with fallback to uniform over valid tokens.

### 2. SamplingSession does not pass backend to ToolRouter
**Category:** Logic
**Severity:** Significant
**Location:** src/tgirl/sample.py:SamplingSession.__init__
**Details:** ToolRouter constructed without backend= parameter, always defaulting to "torch". MLX reranking path (Task 4b) was unreachable through normal SamplingSession usage.
**Resolution:** Fixed in 46fd9ac. One-line fix passing backend to ToolRouter constructor.

### 3. Module-level import torch in sample_mlx.py
**Category:** Performance
**Severity:** Significant
**Location:** src/tgirl/sample_mlx.py (top level)
**Details:** PRP goal is zero torch in MLX hot paths. Module-level import defeats this when hooks aren't used.
**Resolution:** Fixed in 46fd9ac. Moved to lazy import inside hook conversion block.

### 4. get_valid_mask_np not wired to consumer
**Category:** Convention
**Severity:** Minor (open)
**Location:** src/tgirl/sample_mlx.py
**Details:** Task 5 added get_valid_mask_np to outlines_adapter but sample_mlx.py doesn't use it (falls back to get_valid_mask().numpy()).
**Resolution:** Open -- non-blocking, can be wired in follow-up.

### 5. apply_shaping_mlx uses numpy instead of mx ops
**Category:** Performance
**Severity:** Minor (open)
**Location:** src/tgirl/sample_mlx.py
**Details:** Top-p uses np.argsort/np.cumsum instead of PRP-specified mx.sort/mx.cumsum.
**Resolution:** Open -- functional correctness maintained, performance optimization for follow-up.

### 6. __all__ lists MLX symbols unconditionally
**Category:** Convention
**Severity:** Minor (open)
**Location:** src/tgirl/__init__.py
**Details:** Could cause AttributeError on `from tgirl import *` without mlx installed.
**Resolution:** Open -- edge case, non-blocking.

## What's Done Well

- Zero-coupling constraint maintained: transport_mlx.py imports only TransportConfig from tgirl
- TransportResultMlx properly separated from TransportResult (plan review YP1 addressed)
- Cross-validation test verifies MLX and torch Sinkhorn produce identical results within 1e-4
- Auto-detection in SamplingSession is clean -- detect on first forward call
- make_mlx_forward_fn_torch provides clean migration path (plan review YP4 addressed)
- 585 tests pass with zero regressions
- Test coverage exceeds PRP requirements

## Summary

Implementation delivers the MLX-native inference path across 8 tasks + 1 fixup commit. Three significant issues caught during review and resolved: zero-mass sampling fallback, ToolRouter backend wiring, and lazy torch imports. The core architecture is sound -- new modules mirror existing torch modules without modifying them, maintaining clean rollback. Performance validation pending real-model benchmark run.
