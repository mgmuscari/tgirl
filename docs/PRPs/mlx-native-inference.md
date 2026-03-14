# PRP: MLX-Native Inference Path

## Source PRD: docs/PRDs/mlx-native-inference.md
## Date: 2026-03-13

## 1. Context Summary

tgirl's per-token sampling loop converts between MLX and PyTorch on every token (~50ms overhead per token on 0.8B model, vs ~2ms expected). The fix: keep all per-token math in `mx.array` for the MLX path, only converting to Python `int` at final token selection. The torch path stays intact for HF/GGUF backends.

Two new modules mirror the existing torch modules: `sample_mlx.py` (sampling functions) and `transport_mlx.py` (Sinkhorn OT). The `SamplingSession` dispatches to the correct backend based on the forward_fn return type.

## 2. Codebase Analysis

### Existing patterns
- `transport.py` (346 lines): Zero-coupling module. Contains `TransportConfig`, `TransportResult`, `_check_bypass`, `_standard_masking`, `_compute_cost_submatrix`, `_sinkhorn_log_domain`, `_apply_transport_plan`, `redistribute_logits`. All torch tensors.
- `sample.py` (739 lines): `GrammarState` protocol (line 37), `InferenceHook` protocol (line 46), `apply_penalties` (line 100), `apply_shaping` (line 148), `run_constrained_generation` (line 234), `SamplingSession` (line 378). All torch.
- `cache.py` (212 lines): `make_mlx_forward_fn` currently does MLX forward + converts to torch via `_mlx_to_torch`. `make_hf_forward_fn` stays torch-native.
- `outlines_adapter.py` (112 lines): `LLGuidanceGrammarState.get_valid_mask()` returns `torch.Tensor` via llguidance's torch bitmask API.
- `rerank.py` (156 lines): `ToolRouter.route()` calls `run_constrained_generation` -- needs MLX dispatch too.

### Key conversion boundaries
1. **Forward fn** (`cache.py:71-75`): Currently `mlx_logits -> .astype(mx.float32) -> mx_eval -> np.array -> torch.from_numpy`. Target: return `mx.array` directly.
2. **Grammar mask** (`outlines_adapter.py:52-73`): llguidance produces torch bitmask. Must convert to `mx.array` once per token.
3. **Embeddings** (`benchmarks/run_bfcl.py:129-132`): Currently extracted from MLX model as `mx.array -> np.array -> torch.from_numpy`. For MLX path, keep as `mx.array`.
4. **Token sampling** (`sample.py:310-320`): `torch.multinomial(probs, 1)`. MLX equivalent: `mx.random.categorical(logits)` (takes logits, not probs).
5. **OT redistribution** (`transport.py:223-345`): Full Sinkhorn in torch. MLX version mirrors it.

### Conventions
- Modules follow zero-coupling principle (transport has no tgirl deps)
- Lazy imports for optional dependencies (`import mlx.core as mx` inside functions)
- `structlog` for logging
- Pydantic `BaseModel` with `frozen=True` for configs/results
- Tests use mock models that record calls

## 3. Implementation Plan

**Test Command:** `pytest tests/test_transport_mlx.py tests/test_sample_mlx.py tests/test_cache.py -v`

### Task 1: Create `src/tgirl/transport_mlx.py` -- MLX-native Sinkhorn OT

**Files:** `src/tgirl/transport_mlx.py` (CREATE), `tests/test_transport_mlx.py` (CREATE)

**Approach:**
Port all functions from `transport.py` to MLX equivalents, maintaining zero coupling to other tgirl modules. The module mirrors `transport.py` exactly:
- `_check_bypass_mlx(logits: mx.array, valid_mask: mx.array, config) -> (bool, str|None)` -- uses `mx.softmax`, `mx.sum`
- `_standard_masking_mlx(logits: mx.array, valid_mask: mx.array) -> mx.array` -- uses `mx.where`
- `_compute_cost_submatrix_mlx(embeddings: mx.array, invalid_indices, valid_indices) -> mx.array` -- cosine distance via normalized matmul
- `_sinkhorn_log_domain_mlx(cost, source, target, epsilon, max_iter, threshold) -> (plan, wasserstein, iters)` -- log-domain Sinkhorn with `mx.logsumexp`
- `_apply_transport_plan_mlx(plan, valid_indices, original_logits, vocab_size) -> mx.array`
- `redistribute_logits_mlx(logits: mx.array, valid_mask: mx.array, embeddings: mx.array, config=None) -> TransportResultMlx` -- public API, returns MLX-native result type

Define `TransportResultMlx(NamedTuple)` in `transport_mlx.py` with `logits: mx.array` instead of `torch.Tensor`. This avoids importing `TransportResult` from `transport.py`, which would transitively import torch at module level and defeat the zero-coupling goal. `TransportConfig` is safe to import -- it's a Pydantic model with no torch dependency (only pydantic BaseModel). `TransportResultMlx` mirrors `TransportResult` field-for-field except `logits` is typed `mx.array`.

Lazy `import mlx.core as mx` at module top (the module is only imported on Apple Silicon).

**Tests:**
- `test_bypass_forced_decode_mlx`: 0-1 valid tokens triggers bypass
- `test_bypass_valid_ratio_high_mlx`: >50% valid triggers bypass
- `test_bypass_invalid_mass_low_mlx`: <1% invalid mass triggers bypass
- `test_standard_masking_mlx`: invalid positions set to -inf
- `test_sinkhorn_convergence_mlx`: plan marginals match source/target within tolerance
- `test_sinkhorn_matches_torch`: MLX and torch Sinkhorn produce plans within 1e-4 on identical inputs (cross-validation)
- `test_redistribute_full_path_mlx`: end-to-end redistribution produces valid logits
- `test_zero_coupling`: module imports only `TransportConfig` from tgirl, nothing else (no torch transitive import)

**Validation:**
```bash
pytest tests/test_transport_mlx.py -v
ruff check src/tgirl/transport_mlx.py
```

### Task 2: Create `src/tgirl/sample_mlx.py` -- MLX-native sampling functions

**Files:** `src/tgirl/sample_mlx.py` (CREATE), `tests/test_sample_mlx.py` (CREATE)

**Approach:**
Port `apply_penalties`, `apply_shaping`, and `run_constrained_generation` from `sample.py` to MLX:

- `apply_penalties_mlx(logits: mx.array, intervention: ModelIntervention, token_history: list[int]) -> mx.array` -- same logic, `mx.array` indexing for penalty application
- `apply_shaping_mlx(logits: mx.array, intervention: ModelIntervention) -> mx.array` -- temperature division, top-k via `mx.topk`, top-p via `mx.sort`/`mx.cumsum`/`mx.softmax`
- `run_constrained_generation_mlx(grammar_state, forward_fn, tokenizer_decode, embeddings: mx.array, hooks, transport_config, max_tokens, context_tokens) -> ConstrainedGenerationResult`
  - Per-token: forward_fn returns `mx.array` logits, grammar mask converted from torch to mx via numpy, penalties/OT/shaping all in mx, sampling via `mx.random.categorical(logits)` (note: takes logits not probs)
  - Returns same `ConstrainedGenerationResult` (tokens are plain `list[int]`, telemetry fields are plain `list[float]`)

Grammar mask conversion: call `grammar_state.get_valid_mask(vocab_size)` which returns `torch.Tensor`, then `mx.array(mask.numpy())` to get `mx.array`. This is one torch->numpy->mx hop per token but it's a bool vector, not a float computation.

Hook interface: `InferenceHook.pre_forward` receives `logits: torch.Tensor` in the current protocol. For the MLX path, we convert `mx.array` logits to torch only for the hook call, since hooks are user-provided and may depend on torch. The hook returns `ModelIntervention` (plain data), so no conversion needed on the output.

**Important:** `GrammarTemperatureHook.pre_forward` (`sample.py:83-97`) internally calls `grammar_state.get_valid_mask(vocab_size)`, which produces a `torch.Tensor`. This means the MLX constrained loop will see the grammar mask fetched twice per token — once by `run_constrained_generation_mlx` (converted to `mx.array`), and once by the hook itself (stays torch). To avoid this double-conversion overhead, `run_constrained_generation_mlx` should pass the already-computed `mx.array` valid mask to hooks via an extended hook interface, or provide a `GrammarTemperatureHookMlx` that accepts `mx.array` logits and valid mask directly. The simplest v1 approach: `run_constrained_generation_mlx` pre-computes the valid mask once, converts it to both `mx.array` (for OT/sampling) and caches the torch version (for hook calls). The hook's internal `get_valid_mask` call is then redundant but harmless — it returns the same torch tensor from llguidance's internal state. If profiling shows this is a bottleneck, introduce `InferenceHookMlx` protocol in a follow-up.

**Tests:**
- `test_apply_penalties_mlx_repetition`: repetition penalty modifies correct positions
- `test_apply_penalties_mlx_no_penalties`: no-op when all None
- `test_apply_shaping_mlx_temperature`: temperature division correct
- `test_apply_shaping_mlx_top_k`: top-k masking correct
- `test_apply_shaping_mlx_top_p`: top-p nucleus sampling correct
- `test_apply_shaping_mlx_greedy`: temperature=0 selects argmax
- `test_run_constrained_generation_mlx_basic`: mock model, mock grammar, produces tokens
- `test_run_constrained_generation_mlx_accepts`: stops when grammar accepts
- `test_run_constrained_generation_mlx_max_tokens`: stops at max_tokens

**Validation:**
```bash
pytest tests/test_sample_mlx.py -v
ruff check src/tgirl/sample_mlx.py
```

### Task 3: Modify `cache.py` -- MLX forward returns `mx.array`

**Files:** `src/tgirl/cache.py` (MODIFY), `tests/test_cache.py` (MODIFY)

**Approach:**
Change `make_mlx_forward_fn` to return raw `mx.array` logits instead of converting to torch:
- Remove `_mlx_to_torch` helper
- Remove `import torch` from module top (it's still needed for `make_hf_forward_fn`)
- Forward closure returns `mlx_logits[0, -1, :].astype(mx.float32)` directly (no numpy/torch conversion)
- Remove `mx_eval` call -- let MLX handle lazy graph materialization naturally. Only call `mx_eval` if the consumer needs the values immediately (the sampling loop will call it when sampling)
- Update type hint: `Callable[[list[int]], Any]` since return type is now backend-specific

Rename the current torch-returning function to `make_mlx_forward_fn_torch` (preserves existing behavior for downstream users who depend on `torch.Tensor` output). The new `make_mlx_forward_fn` returns `mx.array` natively. Both are exported from `__init__.py`. This breaks existing imports loudly (NameError on the old name if they don't update) rather than silently returning wrong types. The `SamplingSession` dispatch handles both return types transparently.

Migration path: callers using `make_mlx_forward_fn` who pass it to `SamplingSession` need no changes (session auto-detects). Callers who directly consume the `torch.Tensor` return should switch to `make_mlx_forward_fn_torch`.

**Tests:**
- Update existing `TestMakeMlxForwardFn` tests to expect `mx.array` output instead of `torch.Tensor`
- `test_mlx_forward_fn_returns_mx_array`: verify return type is `mx.array`
- `test_mlx_forward_fn_no_torch_conversion`: verify no torch import in the hot path
- Keep `TestMakeHFForwardFn` tests unchanged (still torch)

**Validation:**
```bash
pytest tests/test_cache.py -v
ruff check src/tgirl/cache.py
```

### Task 4: Modify `sample.py` -- Backend dispatch in SamplingSession

**Files:** `src/tgirl/sample.py` (MODIFY), `tests/test_sample.py` (MODIFY -- add dispatch tests only)

**Approach:**
Add backend detection and dispatch to `SamplingSession`:

1. In `__init__`, probe `forward_fn` return type is not practical (would require a dummy call). Instead, add an optional `backend: Literal["torch", "mlx", "auto"] = "auto"` parameter to `SamplingSession.__init__`. When `"auto"`, detect on first forward call.

2. In `run()`, the freeform loop and constrained generation dispatch:
   - `"torch"` path: current code unchanged
   - `"mlx"` path: freeform sampling uses `mx.softmax`, `mx.random.categorical`; constrained generation calls `run_constrained_generation_mlx` from `sample_mlx.py`

3. For freeform MLX path in `run()`: instead of extracting to a separate function, add an `_is_mlx` flag set on first forward call. If `_is_mlx`:
   - `raw_logits` is `mx.array` -- divide by temperature directly
   - Use `mx.softmax` + `mx.random.categorical` for sampling
   - Token ID extraction: `int(token_id.item())`

4. For constrained path: call `run_constrained_generation_mlx` with `mx.array` embeddings (converted once at session start).

5. Embeddings: if `_is_mlx` and embeddings are `torch.Tensor`, convert once: `mx.array(embeddings.numpy())`. Store as `self._embeddings_mlx`.

6. Reranking: `SamplingSession` passes `backend` flag to `ToolRouter` (see Task 4b).

**Tests:**
- `test_session_torch_backend`: existing behavior unchanged
- `test_session_mlx_backend_detection`: forward_fn returning mx.array triggers MLX path
- `test_session_explicit_backend`: `backend="mlx"` forces MLX path
- `test_session_freeform_mlx`: freeform generation works with mx.array forward_fn

**Validation:**
```bash
pytest tests/test_sample.py -v
pytest tests/test_sample_mlx.py -v
```

### Task 4b: MLX-native reranking in ToolRouter

**Files:** `src/tgirl/rerank.py` (MODIFY), `tests/test_rerank.py` (MODIFY)

**Approach:**
`ToolRouter.route()` (`rerank.py:120`) calls `run_constrained_generation` and stores `self._embeddings` as `torch.Tensor`. This is a full constrained generation pass that needs the MLX path for consistency. Changes:

1. Add `backend: Literal["torch", "mlx"] = "torch"` parameter to `ToolRouter.__init__`.
2. Store embeddings in both formats: `self._embeddings` (torch, existing) and `self._embeddings_mlx` (mx.array, lazy-converted on first MLX route call).
3. In `route()`, dispatch to `run_constrained_generation_mlx` when `self._backend == "mlx"`, passing `self._embeddings_mlx`.
4. `SamplingSession` passes its detected backend to `ToolRouter` at construction time.

**Tests:**
- `test_router_torch_path_unchanged`: existing routing tests still pass
- `test_router_mlx_path`: routing with MLX forward_fn and mx.array embeddings
- `test_router_mlx_embeddings_lazy_conversion`: torch embeddings converted to mx.array only on first MLX route call

**Validation:**
```bash
pytest tests/test_rerank.py -v
```

### Task 5: Update `outlines_adapter.py` -- numpy mask accessor

**Files:** `src/tgirl/outlines_adapter.py` (MODIFY)

**Approach:**
Add `get_valid_mask_np(vocab_size) -> np.ndarray` to `LLGuidanceGrammarState`. This returns the boolean mask as a numpy array, which is cheaper to convert to `mx.array` than going torch -> numpy -> mx.

Implementation: llguidance's bitmask is applied to a torch zeros tensor, then compared to -inf. The numpy version does the same but with a numpy array: `np.zeros(vocab_size)` -> apply bitmask -> `mask > -inf`. If llguidance doesn't support numpy directly, convert from torch: `mask.numpy()`.

The `sample_mlx.py` constrained generation calls `get_valid_mask_np` if available (via `hasattr` check), falling back to `get_valid_mask(...).numpy()`.

**Tests:**
- `test_get_valid_mask_np_returns_ndarray`: verify return type
- `test_get_valid_mask_np_matches_torch`: numpy mask matches torch mask values

**Validation:**
```bash
pytest tests/test_outlines_adapter.py -v
```

### Task 6: Update benchmarks and examples

**Files:** `benchmarks/run_bfcl.py` (MODIFY), `examples/showcase_unified_api.py` (MODIFY)

**Approach:**
The benchmarks/examples already use `make_mlx_forward_fn`. After Task 3, this returns `mx.array` instead of torch. The `SamplingSession` auto-detects MLX backend. Changes needed:
- Remove explicit `torch.from_numpy(np.array(mlx_embed))` for embeddings -- pass MLX embeddings directly, let session convert if needed
- Or: keep embeddings as torch (SamplingSession converts internally). Simpler -- less change to call sites.
- Add backend stats logging (which path was used)

**Tests:** Integration test via benchmark run (not unit test).

**Validation:**
```bash
PYTHONUNBUFFERED=1 python -u benchmarks/run_bfcl.py \
    --model mlx-community/Qwen3.5-0.8B-MLX-4bit \
    --category simple_python --limit 5
# Expect: <5ms/token avg, <5s per entry
```

### Task 7: Exports and wiring

**Files:** `src/tgirl/__init__.py` (MODIFY)

**Approach:**
Export new public APIs:
- `from tgirl.transport_mlx import redistribute_logits_mlx, TransportResultMlx`
- `from tgirl.sample_mlx import apply_penalties_mlx, apply_shaping_mlx, run_constrained_generation_mlx`
- `from tgirl.cache import make_mlx_forward_fn_torch` (compatibility alias for torch-returning wrapper)

Guard with try/except ImportError since mlx is optional.

**Validation:**
```bash
pytest tests/ -v
ruff check src/tgirl/
```

## 4. Validation Gates
```bash
# Lint + type check
ruff check src/tgirl/transport_mlx.py src/tgirl/sample_mlx.py src/tgirl/cache.py
mypy src/tgirl/transport_mlx.py src/tgirl/sample_mlx.py

# Unit tests (new)
pytest tests/test_transport_mlx.py tests/test_sample_mlx.py -v

# All tests (regression)
pytest tests/ -v

# Performance validation
PYTHONUNBUFFERED=1 python -u benchmarks/run_bfcl.py \
    --model mlx-community/Qwen3.5-0.8B-MLX-4bit \
    --category simple_python --limit 5
```

## 5. Rollback Plan

All new code is in new modules (`transport_mlx.py`, `sample_mlx.py`). The torch path is unchanged. Rollback:
1. Revert `cache.py` `make_mlx_forward_fn` to return torch tensors
2. Revert `sample.py` backend dispatch (remove MLX branch)
3. Delete `transport_mlx.py` and `sample_mlx.py`
4. Existing tests verify torch path still works

## 6. Uncertainty Log

- **`mx.random.categorical` API**: Takes logits (not probabilities). Need to verify it handles -inf logits correctly (masked positions). If not, manual sampling via `mx.softmax` + cumulative sum + `mx.random.uniform` comparison.
- **`mx.logsumexp` availability**: Assumed available in MLX. If not, implement as `mx.log(mx.sum(mx.exp(x), axis=...))` with max-subtraction for stability.
- **llguidance numpy support**: Assumed llguidance's `apply_token_bitmask_inplace` only works with torch tensors. If it supports numpy, Task 5 simplifies significantly.
- **`TransportResult` type mismatch**: RESOLVED — `TransportResultMlx(NamedTuple)` defined in `transport_mlx.py` with `logits: mx.array`. Only `TransportConfig` is imported from `transport.py` (no torch transitive dependency). Consumers in the MLX path use `TransportResultMlx`; the torch path continues using `TransportResult`.
- **Hook compatibility**: PARTIALLY RESOLVED — `GrammarTemperatureHook` calls `grammar_state.get_valid_mask()` internally, causing a redundant torch mask computation in the MLX path. The v1 approach caches both mx and torch masks per token. If profiling shows this is a bottleneck, introduce `InferenceHookMlx` protocol that receives `mx.array` logits and pre-computed mask.
- **Reranking in MLX path**: `ToolRouter.route()` calls `run_constrained_generation`. For full MLX path, this needs to call `run_constrained_generation_mlx` instead. The router currently stores embeddings as torch -- needs MLX embeddings for the MLX path.
