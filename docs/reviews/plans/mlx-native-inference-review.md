# Plan Review: mlx-native-inference

## Verdict: APPROVED
## Reviewer Stance: Team -- Senior Training Partner + Proposer
## Date: 2026-03-13
## Mode: Agent Team (concurrent review + revision, interrupted by quota, completed by team lead)

## Yield Points Found

### 1. TransportResult NamedTuple breaks with mx.array
**Severity:** Structural (HIGH)
**Evidence:** `TransportResult` is a `NamedTuple` with `logits: torch.Tensor` at `transport.py:35-46`. Importing it from `transport.py` transitively imports torch at module level, defeating zero-coupling. Duck-typing claim is incorrect for NamedTuples.
**Proposer Response:** Accepted. Defined `TransportResultMlx(NamedTuple)` in `transport_mlx.py` with `logits: mx.array`. Only `TransportConfig` imported from `transport.py`. Zero-coupling test updated.
**PRP Updated:** Yes (Task 1 approach, tests, Task 7 exports, Uncertainty Log)

### 2. GrammarTemperatureHook calls get_valid_mask internally, creating hidden double-conversion
**Severity:** Structural (HIGH)
**Evidence:** `GrammarTemperatureHook.pre_forward` at `sample.py:83-97` calls `grammar_state.get_valid_mask(vocab_size)` internally. In the MLX constrained loop, the grammar mask is already computed and converted to `mx.array` -- the hook's internal call produces a redundant torch tensor.
**Proposer Response:** Accepted (team lead). v1 approach: cache both mx and torch masks per token. Hook's redundant `get_valid_mask` call is harmless (returns same result from llguidance state) but wastes a torch allocation. `InferenceHookMlx` deferred to follow-up if profiling warrants.
**PRP Updated:** Yes (Task 2 hook interface section, Uncertainty Log)

### 3. ToolRouter.route() hardcoded to torch path, Task 4 underspecifies the fix
**Severity:** Moderate (MEDIUM)
**Evidence:** `rerank.py:120` calls `run_constrained_generation` (torch-only). `self._embeddings` stored as `torch.Tensor` at line 31. PRP Task 4 line 156 mentions it in one line with no concrete approach.
**Proposer Response:** Accepted (team lead). Added new Task 4b with full approach: backend parameter on ToolRouter, lazy mx.array embedding conversion, dispatch to `run_constrained_generation_mlx`, dedicated tests.
**PRP Updated:** Yes (new Task 4b added)

### 4. Task 3 breaks public API of make_mlx_forward_fn with no migration path
**Severity:** Moderate (MEDIUM)
**Evidence:** `make_mlx_forward_fn` is exported in `__init__.py` and is part of the public API. Changing return type from `torch.Tensor` to `mx.array` breaks downstream silently.
**Proposer Response:** Accepted (team lead). Renamed current function to `make_mlx_forward_fn_torch` (compatibility). New `make_mlx_forward_fn` returns `mx.array`. Both exported. Migration path documented.
**PRP Updated:** Yes (Task 3 approach, Task 7 exports)

## What Holds Well

- Correct identification of the conversion boundary (llguidance returns torch, convert once per token as bool vector)
- Clean separation: new modules mirror existing ones rather than modifying core paths
- Zero-coupling constraint respected for transport_mlx.py
- Uncertainty log is honest and comprehensive
- Rollback plan is clean -- all new code in new modules, torch path unchanged
- `TransportConfig` reuse is safe (plain Pydantic model, no torch dependency)

## Summary

All 4 yield points (2 structural, 2 moderate) have been addressed in the PRP. The plan is architecturally sound with clear module boundaries and a clean rollback path. The hook double-conversion in YP2 is accepted as a known overhead for v1 with a documented follow-up path. The PRP now has 8 tasks (original 7 + new Task 4b for reranking).
