# Code Review: constrained-sampling-engine

## Verdict: APPROVED
## Reviewer Stance: Team — Proposer + Code Review Partner
## Date: 2026-03-12
## Mode: Agent Team (message-gated incremental review)

## PRP Compliance

| Task | Commit | Status | Notes |
|------|--------|--------|-------|
| 1: SessionConfig + ModelIntervention | `7ae774e` | Implemented as specified | |
| 2: InferenceHook + merge_interventions | `14723db` | Implemented as specified | |
| 3: GrammarTemperatureHook | `a07b708` | Implemented as specified | |
| 4: apply_penalties + apply_shaping | `3698e65` | Implemented as specified | Pre/post-OT split per plan review |
| 5: run_constrained_generation | `db388c5` | Implemented as specified | 9-step pipeline matches PRP |
| 6: DelimiterDetector | `384c3e0` | Implemented as specified | Bounded sliding window per plan review |
| 7: SamplingSession | `e436728` + fix `52bb7af` | Implemented after fix | Quota reduction + telemetry fixed |
| 8: __init__.py + integration | `3322e68` | Implemented as specified | |

## Issues Found

### 1. Cross-cycle quota reduction not implemented
**Category:** Logic / Spec Mismatch
**Severity:** Blocking
**Location:** `src/tgirl/sample.py:462-464`
**Details:** `_snapshot_with_remaining_quotas()` from PRP was absent. `self._registry.snapshot()` called directly, returning full quotas every cycle. Breaks TGIRL.md 3.3 "safety by construction" — exhausted tools remained expressible.
**Resolution:** Fixed in `52bb7af`. Method added at `sample.py:580-598`, called at line 466. Two tests added.

### 2. Telemetry type mismatch and never populated
**Category:** Spec Mismatch
**Severity:** Blocking
**Location:** `src/tgirl/sample.py:364, 408`
**Details:** `SamplingResult.telemetry` typed as `list[dict[str, Any]]` (PRP specifies `list[TelemetryRecord]`), never populated.
**Resolution:** Fixed in `52bb7af`. Type corrected, `TelemetryRecord` constructed per cycle with all fields.

### 3. Missing PRP test cases
**Category:** Test Quality
**Severity:** Significant
**Location:** `tests/test_sample.py`
**Details:** No session_timeout test, no two-cycle quota tracking test, weak quotas_consumed assertion.
**Resolution:** Fixed in `52bb7af`. Timeout test added, quota reduction tests added.

### 4. Bare symbol counting in threading position
**Category:** Logic
**Severity:** Significant
**Location:** `src/tgirl/sample.py:553`
**Details:** Regex `\(\s*{name}\b` only matched parenthesized calls. Bare `bar` in `(-> (foo x) bar)` not counted.
**Resolution:** Fixed in `52bb7af`. Regex updated with lookbehind for whitespace-preceded bare symbols. Test added.

### 5. Top-k tie behavior
**Category:** Convention
**Severity:** Minor (noted)
**Location:** `src/tgirl/sample.py` (apply_shaping)
**Details:** Threshold-based top-k filtering keeps >k tokens when values are tied. Standard industry behavior, not blocking.
**Resolution:** Noted, not changed. Consistent with HuggingFace.

### 6. Telemetry sentinel values
**Category:** Convention
**Severity:** Minor
**Location:** `src/tgirl/sample.py` (run_constrained_generation)
**Details:** Sentinel -1.0 for unset temperature/top_p in telemetry arrays. Consumers need to know this convention.
**Resolution:** Noted, not changed.

### 7. Inline imports inside loop body
**Category:** Convention
**Severity:** Nit
**Location:** `src/tgirl/sample.py` (multiple locations)
**Details:** `import hashlib`, `from tgirl.grammar import generate`, `from tgirl.compile import run_pipeline` inside method bodies.
**Resolution:** Noted, not changed. Functional pattern, avoids circular imports.

## What's Done Well

- **Protocol-based abstractions** (`GrammarState`, `InferenceHook`) cleanly decouple from Outlines — future backend swaps won't require changes to the sampling loop
- **Pre/post-OT split** (apply_penalties + apply_shaping) correctly separates concerns per plan review feedback
- **Bounded delimiter detection** with O(1) memory sliding window
- **Defensive probability normalization** with fallback to uniform distribution over valid tokens
- **400 tests passing** across the full suite with zero regressions
- **Proposer fixed all blocking issues promptly** once notified (message delivery problem caused initial miss)

## Summary

All 8 PRP tasks implemented across 9 commits (8 task commits + 1 fix commit). Two blocking issues were caught by incremental review — cross-cycle quota reduction (safety invariant) and telemetry type mismatch — both resolved in fix commit `52bb7af`. 400 tests passing, ruff clean. The implementation faithfully follows the PRP specification with all plan review yield points incorporated.

**APPROVED for security audit.**
