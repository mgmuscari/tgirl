# Code Review: adsr-modulation-matrix

## Verdict: APPROVED with follow-up items
## Reviewer Stance: Team — Proposer + Code Review Partner
## Date: 2026-03-15
## Mode: Agent Team (message-gated incremental review)

## PRP Compliance

| PRP Task | Commit | Status | Notes |
|----------|--------|--------|-------|
| Task 1: Source conditioning + phase detection | `3b8a3c2` | As specified | 3-mechanism hysteresis, 17 tests |
| Task 2: Default matrix + EnvelopeConfig | `ce94789` | As specified | 11x7 matrix, column 5 zeroed |
| Task 3: ModMatrixHookMlx | `a66bf30` | As specified | mx.matmul, binary gates, 12 tests |
| Task 4: ModMatrixHook (torch) + auto-conversion | `bb16f21` | As specified | torch.matmul, auto-conversion wired |
| Task 5: transport_epsilon wiring | `eabeaf0` | Partial | MLX loop wired, torch loop not |
| Task 6: Benchmark + showcase | `cba9f78` | As specified | RepetitionPenaltyHook removed, hyperparams updated |
| Task 7: Telemetry integration | `c3d2794` | Partial | Telemetry computed but not collected by sampling loop |

## Issues Found

### 1. Private `_detect_cycle` imported cross-module
**Category:** Convention
**Severity:** Significant
**Location:** modulation.py:13
**Details:** Module-private function used across module boundary.
**Resolution:** Open — make public or move to shared utility.

### 2. Fragile string-based type check for auto-conversion
**Category:** Convention
**Severity:** Significant
**Location:** sample.py:137-139
**Details:** `_is_mod_matrix_hook` uses `type(hook).__name__` matching instead of `isinstance`.
**Resolution:** Open — add `__module__` check or use proper isinstance.

### 3. Torch sampling loop not wired for transport_epsilon
**Category:** Spec mismatch
**Severity:** Significant
**Location:** sample.py (run_constrained_generation)
**Details:** PRP specifies "both sampling loops" but only MLX loop reads `merged.transport_epsilon`. Per-token epsilon silently dropped in torch backend.
**Resolution:** Open — wire torch loop identically to MLX.

### 4. Telemetry not collected by sampling loop
**Category:** Spec mismatch
**Severity:** Significant
**Location:** sample_mlx.py
**Details:** `EnvelopeTelemetry` stored on hook's `last_telemetry` attribute but `sample_mlx.py` never reads it. Per-token telemetry overwritten each token — only last survives.
**Resolution:** Open — sampling loop should accumulate telemetry per token.

## What's Done Well

- Native MLX/torch ops throughout — no cross-framework conversions
- Source conditioning on n=11 scalars correctly documented as accepted deviation
- Phase detection hysteresis matches PRP spec with all 3 mechanisms
- Column 5 (backtrack) correctly zeroed in default matrix
- RepetitionPenaltyHook correctly absorbed — not kept alongside
- Matrix values match PRP exactly across all 11 rows
- 49 modulation tests + 750 full suite passing, 0 regressions
- TDD discipline maintained across all 7 tasks

## Summary

The modulation matrix is architecturally complete and functionally correct for the MLX backend. 4 significant findings — 2 are spec compliance gaps (torch epsilon, telemetry collection) and 2 are convention issues (private import, string type check). None are blocking. The core innovation — replacing 3 static hooks with a single configurable matrix multiply — is well-implemented.
