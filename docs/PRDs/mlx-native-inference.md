# PRD: MLX-Native Inference Path

## Status: IMPLEMENTED
## Author: agent
## Date: 2026-03-13
## Branch: feature/mlx-native-inference

## 1. Problem Statement

tgirl's sampling pipeline operates entirely in PyTorch tensors. When running on Apple Silicon with MLX models, every token requires a round-trip conversion: `mx.array` to numpy to `torch.from_numpy()`. This conversion happens on every forward pass, and the downstream pipeline (penalties, OT redistribution, shaping, sampling) also operates in torch before discarding the result as a scalar `int` token ID.

With KV caching (merged in `feature/bfcl-benchmark-integration`), the model forward pass itself is fast (~2ms for a 0.8B model), but the per-token overhead from framework conversion and torch tensor operations dominates. Benchmark data shows ~52ms/token on a 0.8B model where ~2-5ms/token is expected. The conversion overhead is the primary suspect.

This affects every MLX user of the library. The torch path must remain intact for HuggingFace/GGUF/other backends -- this is an additive change, not a replacement.

## 2. Proposed Solution

Add an MLX-native code path that keeps all per-token math in `mx.array` from forward pass through sampling, converting to a Python `int` only at the final token selection. Two components:

1. **MLX-native sampling functions** -- `mx.array` versions of `apply_penalties`, `apply_shaping`, freeform sampling, and constrained generation. These live in a new module `src/tgirl/sample_mlx.py` and mirror the torch versions in `sample.py`.

2. **MLX-native Sinkhorn OT** -- Pure MLX implementation of log-domain Sinkhorn, replacing the torch implementation in `transport.py` for the MLX path. Lives in `src/tgirl/transport_mlx.py`. The existing torch transport module stays unchanged (zero-coupling constraint preserved).

The `SamplingSession` detects whether `forward_fn` returns `mx.array` or `torch.Tensor` and dispatches to the appropriate code path. The `GrammarState.get_valid_mask()` boundary is the one conversion point -- llguidance returns torch tensors, which are converted to `mx.array` once per token.

## 3. Architecture Impact

### Files affected
- `src/tgirl/sample_mlx.py` -- NEW: MLX-native sampling functions
- `src/tgirl/transport_mlx.py` -- NEW: MLX-native Sinkhorn OT
- `src/tgirl/cache.py` -- MODIFY: `make_mlx_forward_fn` returns `mx.array` instead of `torch.Tensor`
- `src/tgirl/sample.py` -- MODIFY: `SamplingSession` dispatches to MLX or torch path; `GrammarState` protocol gains optional `get_valid_mask_np()` for backend-agnostic masks
- `src/tgirl/outlines_adapter.py` -- MODIFY: add `get_valid_mask_np()` returning numpy array
- `src/tgirl/__init__.py` -- MODIFY: export new public APIs
- `tests/test_sample_mlx.py` -- NEW: MLX sampling tests
- `tests/test_transport_mlx.py` -- NEW: MLX Sinkhorn tests

### Data model changes
- `forward_fn` return type widens from `torch.Tensor` to `torch.Tensor | mx.array` (the existing contract type hint relaxes, but runtime behavior is compatible)
- `TransportResult` stays unchanged (OT results are scalars + the redistributed logits, which are backend-specific internally but consumed as indexable arrays)

### API changes
- `make_mlx_forward_fn` returns `mx.array` logits instead of `torch.Tensor` (breaking change for callers who inspect the return type, but `SamplingSession` handles both)
- New public functions: `redistribute_logits_mlx()`, `apply_penalties_mlx()`, `apply_shaping_mlx()`

### Dependency additions
- None new. `mlx` is already an optional dependency. The MLX path is gated behind `import mlx` availability.

## 4. Acceptance Criteria

1. MLX-native sampling path produces identical token selections to torch path given the same RNG seed and logits.
2. `make_mlx_forward_fn` returns `mx.array` logits; no torch conversion in the forward closure.
3. `SamplingSession` auto-detects MLX vs torch forward_fn and dispatches accordingly.
4. MLX Sinkhorn produces transport plans within 1e-4 of torch Sinkhorn on identical inputs.
5. Grammar mask conversion (torch to mx) happens exactly once per token, at the `get_valid_mask` boundary.
6. Torch path is completely unchanged -- all existing tests pass unmodified.
7. Per-token overhead on MLX path is <10ms for 0.8B model (vs ~52ms current).
8. `transport_mlx.py` has zero coupling to other tgirl modules (same constraint as `transport.py`).
9. BFCL benchmark runs correctly with the MLX-native path.

## 5. Risk Assessment

- **Grammar mask conversion cost**: llguidance returns torch tensors. Converting to mx.array per token adds overhead. Mitigation: profile to verify it's cheaper than the current full-pipeline torch path. If llguidance supports numpy output, convert from numpy directly (cheaper).
- **Numerical divergence**: MLX and torch may produce slightly different softmax/sampling results due to float32 precision differences. Mitigation: test with tolerance, verify end-to-end behavior matches.
- **MLX random categorical semantics**: May differ from `torch.multinomial` in edge cases (zero probabilities, normalization). Mitigation: test thoroughly.
- **Sinkhorn convergence**: MLX Sinkhorn must converge to the same transport plan as torch version. Mitigation: cross-validate with identical inputs.
- **API breakage**: `make_mlx_forward_fn` return type change could break callers who depend on `torch.Tensor`. Mitigation: document the change; existing callers (benchmark, showcase) go through `SamplingSession` which handles both.

## 6. Open Questions

1. Does llguidance's `fill_next_token_bitmask` / `apply_token_bitmask_inplace` support numpy arrays directly, or only torch tensors? If numpy, we can skip the torch-to-mx hop entirely for grammar masks.
2. Should the MLX path use `mx.random.categorical` (log-probability input) or manual argmax/sampling? Need to verify API compatibility.
3. Should embeddings be stored as `mx.array` for the MLX path? Currently they're torch tensors extracted from the MLX model via numpy -- storing as mx.array avoids another conversion in the OT path.

## 7. Out of Scope

- Modifying the `forward_fn` contract signature (stays `Callable[[list[int]], ...]`)
- GPU/CUDA-specific optimizations
- Replacing llguidance with an MLX-native grammar engine
- Multi-device or distributed inference
- Removing the torch dependency from the library
