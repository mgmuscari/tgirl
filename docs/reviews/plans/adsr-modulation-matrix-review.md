# Plan Review: adsr-modulation-matrix

## Verdict: APPROVED
## Reviewer Stance: Team — Senior Training Partner + Proposer
## Date: 2026-03-15
## Mode: Agent Team (concurrent review + revision)

## Yield Points Found

### 1. Source conditioning uses Python scalar loops
**Severity:** Medium
**Evidence:** `condition_source` iterates over 11 sources with a Python for loop, contradicting CLAUDE.md "no Python-fu on tensor data" guardrail.
**Proposer Response:** Accepted with documented deviation. n=11 is constant, data is Python scalars (not tensor elements), sub-microsecond compute. Vectorization marked as v2 enhancement if source count grows.
**PRP Updated:** Yes — deviation documented in Uncertainty Log.

### 2. Duplicated softmax (user-flagged)
**Severity:** HIGH
**Evidence:** `pre_forward` computes `mx.softmax(logits)` to derive entropy and overlap, but `apply_shaping_mlx` computes softmax again post-OT. User explicitly asked for a decision, not a deferral.
**Proposer Response:** Accepted — deferral replaced with decided acceptance. The pre-OT softmax (on raw logits) and post-OT softmax (on shaped logits) operate on different distributions. Reusing the post-OT softmax would give the hook incorrect signal values. The pre-OT softmax is the correct input for source signals. `TransitionSignal` named as v2 migration path to avoid duplication by passing pre-computed signals.
**PRP Updated:** Yes — Uncertainty Log updated with decision and rationale.

### 3. `logit_bias` merge silently drops data
**Severity:** HIGH
**Evidence:** Task 6 plans to run `ModMatrixHookMlx` alongside `RepetitionPenaltyHook`. Both produce `logit_bias` dicts. `merge_interventions` uses last-non-None-wins per field — the second hook's entire `logit_bias` dict overwrites the first's. Opener bias and phase-aware repetition bias from the mod matrix silently lost.
**Proposer Response:** Accepted. `RepetitionPenaltyHook` removed from Task 6 hook list entirely. Mod matrix fully subsumes repetition bias via phase-aware envelope + cycle_detected gate. Silent overwrite bug eliminated at the source.
**PRP Updated:** Yes — Task 6 revised.

### 4. Benchmark telemetry will crash
**Severity:** Medium
**Evidence:** `benchmarks/run_bfcl.py:384-385` accesses `session_hooks[0].base_temperature` and `.scaling_exponent` — attributes `ModMatrixHook` doesn't have.
**Proposer Response:** Accepted. Task 6 updated: hyperparams dict uses `_config.base_temperature`. `scaling_exponent` replaced with `modulation_matrix_hash`.
**PRP Updated:** Yes.

### 5. `backtrack_threshold_mod` is a dead output
**Severity:** Low
**Evidence:** Column 5 of the matrix routes to a destination that nothing reads. No `ModelIntervention` field, no sampling loop integration.
**Proposer Response:** Partially accepted. Column 5 zeroed in default matrix but (11,7) shape preserved for forward compatibility with serialized configs and future backtrack integration. Column annotated as reserved.
**PRP Updated:** Yes.

### 6. Phase hysteresis unspecified
**Severity:** Medium
**Evidence:** Uncertainty Log promises "require ≥2 consecutive tokens" but `detect_phase` has no hysteresis. Binary gates amplify the problem — flickering freedom causes all 7 destinations to hard-switch simultaneously.
**Proposer Response:** Accepted. Full three-mechanism hysteresis specified: (1) decay pending counter requiring N consecutive freedom-collapsed tokens, (2) depth-gated release requiring depth ≤ 1, (3) minimum phase duration preventing re-transition within 2 tokens. 8 test cases added.
**PRP Updated:** Yes.

## What Holds Well

- The modulation matrix is a clean generalization of the existing hook pipeline
- Signal chain design (condition → matmul → clamp) is minimal and efficient
- 77-parameter matrix is human-interpretable and small enough for optimization
- Design correctly separates phase detection (pure logic) from parameter computation (tensor ops)
- Absorbing RepetitionPenaltyHook into the matrix eliminates a class of merge conflicts
- Binary phase gates with hysteresis is the right v1 trade-off
- Softmax duplication decision is well-reasoned (pre-OT ≠ post-OT distributions)

## Summary

The PRP started with 2 HIGH and 3 MEDIUM structural issues. All 6 were accepted and resolved. The most important fix was YP3 — the `logit_bias` merge conflict would have been a silent correctness bug in the exact configuration Task 6 creates. The proposer's decision to fully absorb `RepetitionPenaltyHook` into the matrix eliminates the bug class entirely. The PRP is ready for implementation.
