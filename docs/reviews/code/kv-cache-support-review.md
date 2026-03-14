# Code Review: kv-cache-support

## Verdict: APPROVED
## Reviewer Stance: Team — Proposer + Code Review Partner
## Date: 2026-03-13
## Mode: Agent Team (message-gated incremental review)

## PRP Compliance

| Task | Spec | Status |
|------|------|--------|
| Task 1: MLX wrapper factory | `make_mlx_forward_fn` with `CacheStats`, prefix detection, cache hit/miss/equal logic | Implemented as specified |
| Task 2: HF wrapper factory | `make_hf_forward_fn` with immutable `past_key_values` pattern | Implemented as specified |
| Task 3: BFCL benchmark update | Replace raw closure with cache wrapper, add stats logging | Implemented as specified (after blocking fix) |
| Task 4: Showcase example update | Same as Task 3 for showcase | Implemented as specified (after blocking fix) |
| Task 5: Exports | Export `CacheStats`, `make_mlx_forward_fn`, `make_hf_forward_fn` | Implemented as specified |

Deviation from spec: `tokenizer` parameter omitted from `make_mlx_forward_fn` signature (spec included it, implementation determined it was unnecessary). Documented as intentional.

## Commits

| Commit | Task | Description |
|--------|------|-------------|
| 68d46c1 | 1 | `CacheStats` + `make_mlx_forward_fn` with KV cache closure |
| 9a31a40 | 2 | `make_hf_forward_fn` with immutable `past_key_values` |
| e773e3e | 3 | BFCL benchmark uses cache wrapper + stats reporting |
| 5c095d6 | 4 | Showcase example uses cache wrapper + stats reporting |
| 20efef1 | 5 | Exports from `tgirl.__init__` |
| 81a4ee9 | fix | Keyword-only `stats`, remove unused logger, TypeError on bad return |
| 9ab6206 | fix | MLX array conversion pipeline (resolves blocker) |

## Issues Found

### 1. MLX array conversion gap
**Category:** Logic
**Severity:** Blocking
**Location:** `src/tgirl/cache.py:make_mlx_forward_fn`
**Details:** Wrapper passed raw Python lists to MLX model and attempted torch-style tensor extraction. Real MLX models return `mx.array` objects requiring materialization, float32 cast, and numpy-to-torch bridging. Mock tests masked this because they returned `torch.Tensor` directly.
**Resolution:** Fixed in 9ab6206. Wrapper now performs full `mx.array([tokens])` input wrapping, logits extraction with float32 cast, graph materialization, and numpy-to-torch conversion.

### 2. API inconsistency — `stats` parameter
**Category:** Convention
**Severity:** Significant
**Location:** `src/tgirl/cache.py:34`
**Details:** MLX wrapper had `stats` as positional parameter while HF wrapper had it as keyword-only. Inconsistent public API.
**Resolution:** Fixed in 81a4ee9. Both wrappers now use keyword-only `stats`.

### 3. Unused logger import
**Category:** Convention
**Severity:** Nit
**Location:** `src/tgirl/cache.py:21`
**Details:** `structlog` logger imported but never used.
**Resolution:** Removed in 81a4ee9.

### 4. Silent type coercion on unknown return types
**Category:** Logic
**Severity:** Nit
**Location:** `src/tgirl/cache.py:103`
**Details:** Fallback branch silently converted unknown types via `torch.tensor()`.
**Resolution:** Replaced with explicit TypeError in 81a4ee9, then replaced with proper MLX conversion in 9ab6206.

### 5. Top-level MLX import in test file
**Category:** Convention
**Severity:** Minor (open)
**Location:** `tests/test_cache.py`
**Details:** Top-level import will fail on non-macOS CI runners. Could use `pytest.importorskip("mlx.core")`.
**Resolution:** Open — non-blocking, can be addressed if CI needs cross-platform support.

### 6. No test for HF `device` parameter propagation
**Category:** Test Quality
**Severity:** Minor (open)
**Location:** `tests/test_cache.py`
**Details:** No test verifies that the `device` kwarg is passed through to `torch.tensor()`.
**Resolution:** Open — non-blocking.

### 7. O(n) defensive copy per call
**Category:** Performance
**Severity:** Minor (open)
**Location:** `src/tgirl/cache.py:105`
**Details:** `list(token_ids)` copies the full token history every forward call.
**Resolution:** Acknowledged, deferred — acceptable for v1.

## What's Done Well

- Clean separation: `cache.py` has zero coupling to other tgirl modules, following the same isolation principle as `transport.py`
- Lazy MLX imports keep the module importable on non-Apple platforms
- `CacheStats` dataclass provides clean observability without log parsing
- Prefix-detection logic is correct and handles all three cases (hit, miss, equal)
- Both wrapper APIs are now consistent after the fixup commit
- Full test suite (541/541) passes throughout

## Summary

Implementation delivers the planned KV cache support across 5 tasks + 2 fixup commits. One blocking issue was caught during review (MLX array conversion gap) and resolved. The core design — closure-based prefix detection with transparent cache management — is sound and delivers the expected speedup for sequential token generation without any changes to the `forward_fn` contract or `sample.py`.
