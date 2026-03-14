# Code Review: psm-deferred

## Verdict: APPROVED
## Reviewer Stance: Team — Proposer + Code Review Partner
## Date: 2026-03-14
## Mode: Agent Team (message-gated incremental review)

## PRP Compliance

| Task | PRP Spec | Status | Commits |
|------|----------|--------|---------|
| 1 | `compute_transition_signal` helper (framework-agnostic) + wire into freeform loop | Implemented as specified, fix for token_log_prob semantics | 8aad9bd, 1af81f5 |
| 2 | `BacktrackSteeringHookMlx` (MLX-native, InferenceHookMlx protocol) | Implemented as specified, clean | 2fc0333 |
| 3 | Best-so-far tracking on Checkpoint (best_tokens, best_mean_log_prob, attempts, with_attempt) | Implemented as specified | 97e7b9f |
| 4 | Full backtrack dispatch loop (torch + MLX) | Implemented as specified, fix for MLX path + divergent token + with_attempt wiring | ff9a06b, 23b37d5 |

All 4 tasks delivered across 6 commits (4 implementation + 2 fix). 693 tests passing (668 original + 25 new). Zero regressions.

## Issues Found

### 1. `token_log_prob` computed as max instead of sampled token (RESOLVED)
**Category:** Logic
**Severity:** Significant
**Location:** `src/tgirl/state_machine.py` (compute_transition_signal)
**Details:** Initial implementation computed `max(log_probs)` instead of PRP-specified "log prob of the sampled token".
**Resolution:** Fixed in 1af81f5. Added `sampled_token_id` parameter. 2 new tests.

### 2. Per-token closure creation in freeform loop (RESOLVED)
**Category:** Performance
**Severity:** Significant
**Location:** `src/tgirl/sample.py` (SamplingSession.run freeform loop)
**Details:** Helper functions for softmax/sum/log were defined inside the per-token loop, creating new closures on every iteration.
**Resolution:** Fixed in 1af81f5. Helpers hoisted outside the loop.

### 3. MLX backtrack dispatch missing (RESOLVED)
**Category:** Logic
**Severity:** Blocking
**Location:** `src/tgirl/sample_mlx.py` (run_constrained_generation_mlx)
**Details:** PRP requires both torch and MLX backends have backtrack support. Initial Task 4 commit only implemented torch.
**Resolution:** Fixed in 23b37d5. Full MLX backtrack dispatch with matching logic, 2 new MLX tests.

### 4. Dead-end token semantics incorrect (RESOLVED)
**Category:** Logic
**Severity:** Significant
**Location:** `src/tgirl/sample.py` (run_constrained_generation backtrack path)
**Details:** Dead-end token used trigger-position token instead of the divergent token at the checkpoint position.
**Resolution:** Fixed in 23b37d5. Now uses `tokens[checkpoint.position]`.

### 5. `grammar_guide_factory` parameter missing (RESOLVED)
**Category:** Logic
**Severity:** Significant
**Location:** `src/tgirl/sample.py`, `src/tgirl/sample_mlx.py`
**Details:** Needed for grammar replay on backtrack but was omitted from function signatures.
**Resolution:** Fixed in 23b37d5. Added to both torch and MLX signatures.

### 6. `with_attempt` not wired — best-so-far tracking was dead code (RESOLVED)
**Category:** Logic
**Severity:** Minor
**Location:** `src/tgirl/sample.py` (run_constrained_generation)
**Details:** Task 3 added `with_attempt()` to Checkpoint but Task 4 never called it.
**Resolution:** Fixed in 23b37d5. Called before backtracking to record best sequence.

### 7. Type annotation inconsistencies (OPEN — minor)
**Category:** Convention
**Severity:** Minor
**Location:** `src/tgirl/sample.py`, `src/tgirl/sample_mlx.py`
**Details:** `confidence_monitor` typed `Any | None` (torch) vs `object | None` (MLX). `backtrack_checkpoint` and `backtrack_events` on ConstrainedGenerationResult typed `Any` instead of `Checkpoint | None` and `list[BacktrackEvent]`.
**Resolution:** Open — suitable for follow-up cleanup. Does not affect runtime behavior.

### 8. Freeform grammar_mask_overlap always 0.0 (KNOWN LIMITATION)
**Category:** Logic
**Severity:** Minor
**Location:** `src/tgirl/sample.py` (SamplingSession.run freeform loop)
**Details:** No grammar state exists during freeform generation, so grammar_mask_overlap and grammar_freedom are always 0.0. ConfidenceTransitionPolicy relies primarily on w_certainty and w_quality signals in freeform mode.
**Resolution:** Known limitation. Speculative grammar state construction not in PRP scope.

## What's Done Well

- **Framework-agnostic compute_transition_signal** — accepts callable softmax/sum/log, zero framework imports in state_machine.py
- **Parity between torch and MLX** — both backends have full backtrack support with matching logic
- **BacktrackSteeringHookMlx** properly conforms to InferenceHookMlx protocol (valid_mask instead of grammar_state)
- **Backward-compatible defaults** — all new fields on ConstrainedGenerationResult and Checkpoint have defaults, existing code unaffected
- **Review-driven quality** — 2 fix commits resolved all blocking and significant findings. The review caught real issues (MLX path omission, divergent token semantics, dead code).

## Summary

All 4 deferred items from the initial PSM implementation are complete. The compute_transition_signal helper enables ConfidenceTransitionPolicy to receive real model signals. BacktrackSteeringHookMlx provides MLX parity. Best-so-far tracking is wired into the backtrack flow. The full backtrack dispatch loop works on both torch and MLX backends. 693 tests pass with zero regressions. Minor type annotation cleanup remains as follow-up.
