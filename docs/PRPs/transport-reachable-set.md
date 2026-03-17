# PRP: Transport Reachable Set Optimization

## Source PRD: docs/PRDs/transport-reachable-set.md
## Date: 2026-03-15

## 1. Context Summary

Optimal transport (Sinkhorn) is bypassed on ~80% of constrained generation tokens because the cost matrix `(n_invalid x n_valid)` exceeds `max_problem_size` (1M elements). With n_invalid ~= 248k (full vocabulary minus grammar-valid tokens), even small valid sets produce matrices well over the cap.

We pre-compute the set of token IDs reachable from any grammar terminal at grammar compilation time. This reachable set (expected ~5k-10k tokens) restricts OT to operate on a vastly smaller problem, bringing most grammar positions under the cap and raising OT engagement from ~20% to >80%.

## 2. Codebase Analysis

### Relevant existing patterns

- **Grammar terminal structure** (`src/tgirl/templates/base.cfg.j2`, lines 23-29): Three terminal types to extract:
  - `ESCAPED_STRING` -- imported via `%import common.ESCAPED_STRING`, regex: `/"([^"\\\\]|\\\\.)*"/` (Lark common grammar)
  - `SIGNED_INT: /[+-]?[0-9]{1,18}/` -- inline regex terminal
  - `SIGNED_FLOAT: /[+-]?[0-9]{1,18}(\.[0-9]{1,10})?([eE][+-]?[0-9]{1,3})?/` -- inline regex terminal
  - `SPACE: " "` -- string literal terminal
  - `SYMBOL: /[a-zA-Z_][a-zA-Z0-9_-]*/` -- inline regex terminal (composition.cfg.j2, line 4)
  - String literals in tool/type productions: `"("`, `")"`, tool names, `"True"`, `"False"`, `"nil"`, etc.

- **Grammar output** (`src/tgirl/grammar.py`, lines 45-53): `GrammarOutput` is a frozen Pydantic model with `text`, `productions`, `snapshot_hash`, `tool_quotas`, `cost_remaining`. The `text` field contains the full Lark EBNF grammar. Adding `reachable_tokens` is a backward-compatible field addition.

- **Transport functions** (`src/tgirl/transport.py`, lines 223-345; `src/tgirl/transport_mlx.py`, lines 187-295): Both `redistribute_logits()` and `redistribute_logits_mlx()` accept `logits`, `valid_mask`, `embeddings`, and optional `config`. They compute `invalid_indices = where(~valid_mask)` and `valid_indices = where(valid_mask)` over the full vocabulary. The reachable set optimization replaces this with projection to the reachable subset.

- **Transport zero-coupling invariant** (`CLAUDE.md`, dependency graph): `transport.py` and `transport_mlx.py` must have zero coupling to grammar or registry. The reachable set must be passed as a plain `frozenset[int]`, not a grammar object.

- **Sampling loop wiring** (`src/tgirl/sample.py`, lines 992-1077): `SamplingSession.run()` calls `generate_grammar(snapshot)` to get `grammar_output`, then passes `grammar_output.text` to the grammar guide factory. The constrained generation functions receive `transport_config` but not the grammar output. The reachable set must be threaded through as a new parameter.

- **MLX zero-torch invariant** (`src/tgirl/sample_mlx.py`, module docstring): Zero torch, zero numpy in the hot loop. The reachable set is a Python `frozenset[int]` -- no tensor conversion needed until inside the transport function.

- **Vocab scan pattern** (`src/tgirl/sample_mlx.py`, lines 141-153, `NestingDepthHookMlx.__init__`): Iterating `range(vocab_size)` calling `tokenizer_decode([tid])` is an established pattern. Used for paren-delta pre-computation.

### Conventions to follow

- TDD: RED -> GREEN -> REFACTOR per task
- Frozen Pydantic models for all data structures
- `structlog` for logging
- Native framework ops only (torch for transport.py, MLX for transport_mlx.py)
- No Python list comprehensions on tensor data
- Test class per task, method-level tests

### Integration points

- `grammar.generate()` -> `GrammarOutput` (adds `reachable_tokens`)
- `GrammarOutput.reachable_tokens` -> `run_constrained_generation()` / `run_constrained_generation_mlx()` -> `redistribute_logits()` / `redistribute_logits_mlx()`
- `SamplingSession.run()` wires grammar_output.reachable_tokens through to constrained gen

## 3. Implementation Plan

**Test Command:** `pytest tests/test_reachable_set.py tests/test_transport.py tests/test_transport_mlx.py tests/test_grammar.py -v`

### Task 1: `compute_reachable_set()` -- grammar terminal extraction + vocab scan

**Files:** `src/tgirl/grammar.py`, `tests/test_reachable_set.py` (new)

**Approach:**

Add `compute_reachable_set(grammar_text, tokenizer_decode, vocab_size)` to `grammar.py`. The function:

1. **Extract string literals** from grammar text using regex: find all `"..."` patterns in Lark EBNF. For each literal string, call `tokenizer_decode` on each character to identify matching token IDs. Also tokenize multi-character literals as whole strings (subword tokenizers may merge characters).
   - Pattern: `"([^"\\]*(?:\\.[^"\\]*)*)"` to match Lark double-quoted string literals
   - For each extracted literal, collect the set of characters/substrings that appear

2. **Extract regex terminals** from grammar text: find patterns like `TERMINAL_NAME: /regex/` and `TERMINAL_NAME: /regex/flags`. Known terminals and their regex patterns:
   - `SIGNED_INT: /[+-]?[0-9]{1,18}/`
   - `SIGNED_FLOAT: /[+-]?[0-9]{1,18}(\.[0-9]{1,10})?([eE][+-]?[0-9]{1,3})?/`
   - `SYMBOL: /[a-zA-Z_][a-zA-Z0-9_-]*/`

3. **Resolve imported terminals**: detect `%import common.ESCAPED_STRING` and map to the known regex `/"([^"\\\\]|\\\\.)*"/`.

4. **Scan vocabulary**: for each token ID in `range(vocab_size)`, decode it via `tokenizer_decode([tid])` and check if the decoded string (or any character in it) matches any extracted terminal pattern. Collect matching token IDs into the reachable set.

5. Return `frozenset[int]` of all reachable token IDs.

**Performance note:** The vocab scan is O(vocab_size) with regex matching. Pre-compile all regex patterns. The `NestingDepthHookMlx` constructor uses the same `for tid in range(vocab_size)` pattern and is acceptable at ~1s for 250k tokens.

**Tests (RED first):**

```python
class TestComputeReachableSet:
    def test_returns_frozenset(self):
        """compute_reachable_set returns frozenset[int]."""

    def test_string_literal_tokens_included(self):
        """Tokens matching string literals like '(' ')' 'tool_name' are in the set."""

    def test_digit_tokens_included(self):
        """Tokens matching SIGNED_INT regex (digits, +, -) are in the set."""

    def test_unreachable_tokens_excluded(self):
        """Tokens that cannot match any terminal (e.g., CJK) are excluded."""

    def test_empty_grammar_returns_empty_set(self):
        """Grammar with no terminals produces empty reachable set."""

    def test_deterministic(self):
        """Same inputs produce same output (frozenset equality)."""

    def test_escaped_string_tokens_included(self):
        """Tokens matching ESCAPED_STRING terminal are included."""
```

**Validation:** `pytest tests/test_reachable_set.py::TestComputeReachableSet -v`

---

### Task 2: Add `reachable_tokens` to `GrammarOutput` and wire through `generate()`

**Files:** `src/tgirl/grammar.py`, `tests/test_grammar.py`, `tests/test_reachable_set.py`

**Approach:**

1. Add `reachable_tokens: frozenset[int] | None = None` field to `GrammarOutput` (frozen Pydantic model). Default `None` for backward compatibility.

2. Modify `generate()` to accept an optional `tokenizer_decode: Callable[[list[int]], str] | None` and `vocab_size: int | None` parameters. When both are provided, call `compute_reachable_set(text, tokenizer_decode, vocab_size)` after rendering the grammar and include the result in the returned `GrammarOutput`.

3. When `tokenizer_decode` is not provided (backward compat), `reachable_tokens` remains `None`.

**Tests (RED first):**

```python
class TestGrammarOutputReachableTokens:
    def test_grammar_output_accepts_reachable_tokens(self):
        """GrammarOutput can be constructed with reachable_tokens field."""

    def test_grammar_output_default_none(self):
        """GrammarOutput.reachable_tokens defaults to None."""

    def test_grammar_output_frozen_reachable_tokens(self):
        """reachable_tokens is immutable (frozenset)."""

    def test_generate_without_tokenizer_returns_none(self):
        """generate() without tokenizer_decode sets reachable_tokens=None."""

    def test_generate_with_tokenizer_returns_frozenset(self):
        """generate() with tokenizer_decode populates reachable_tokens."""

    def test_generate_reachable_set_deterministic(self):
        """Same snapshot + tokenizer produces same reachable set."""
```

**Validation:** `pytest tests/test_grammar.py tests/test_reachable_set.py -v`

---

### Task 3: Modify `redistribute_logits_mlx()` to accept and use reachable set

**Files:** `src/tgirl/transport_mlx.py`, `tests/test_transport_mlx.py`

**Approach:**

Add `reachable_tokens: frozenset[int] | None = None` parameter to `redistribute_logits_mlx()`. When provided:

1. **Build reachable index array** once at function entry: `reachable_idx = mx.array(sorted(reachable_tokens))`. This is O(|R|), done once, not per-token.

2. **Project valid_mask to reachable space**: `valid_mask_R = valid_mask[reachable_idx]` -- boolean mask over reachable tokens only.

3. **Project logits to reachable space**: `logits_R = logits[reachable_idx]`.

4. **Run bypass checks and OT on the projected tensors** (same logic, smaller dimensions).

5. **Compute invalid/valid indices within reachable space**: `valid_indices_R = mx.array(np.where(mask_np_R)[0])`, `invalid_indices_R = mx.array(np.where(~mask_np_R)[0])`.

6. **Compute cost submatrix using reachable embeddings**: `embeddings_R = embeddings[reachable_idx]`, then `_compute_cost_submatrix_mlx(embeddings_R, invalid_indices_R, valid_indices_R)`.

7. **Run Sinkhorn on the smaller cost matrix** (same algorithm).

8. **Remap result back to full vocab**: construct full-vocab output tensor (-inf everywhere), scatter valid log-probs back to their original positions using `reachable_idx[valid_indices_R]`.

9. **Zero mass on unreachable tokens**: unreachable tokens get -inf (they're already -inf since we start from a full -inf tensor and only scatter valid positions).

When `reachable_tokens=None`, behavior is identical to current implementation (no projection).

**Key constraint:** All tensor math in `mx.array`. The `frozenset[int] -> sorted list -> mx.array` conversion happens once at function entry, not in a loop. The `np.where` call on the projected mask is the same pattern already used at line 226-228 of `transport_mlx.py`.

**Tests (RED first):**

```python
class TestRedistributeLogitsMlxReachableSet:
    def test_reachable_none_unchanged(self):
        """reachable_tokens=None produces same result as before."""

    def test_reachable_set_enables_ot(self):
        """With reachable set, OT runs where it would have been bypassed."""

    def test_reachable_set_result_shape(self):
        """Output shape is (vocab_size,) regardless of reachable set size."""

    def test_unreachable_tokens_negative_inf(self):
        """Tokens not in reachable set have -inf in output."""

    def test_valid_tokens_have_finite_logits(self):
        """Valid tokens within reachable set have finite log-probs."""

    def test_probabilities_sum_to_one(self):
        """exp(output) sums to ~1.0 over finite values."""

    def test_no_torch_imports(self):
        """Module has zero torch imports (inspect source)."""

    def test_small_reachable_set_runs_ot(self):
        """Reachable set of 100 tokens with 10 valid -> OT runs (1000 < 1M cap)."""

    def test_large_full_vocab_would_bypass(self):
        """Same scenario without reachable set -> OT bypassed (problem_size > 1M)."""
```

**Validation:** `pytest tests/test_transport_mlx.py::TestRedistributeLogitsMlxReachableSet -v`

---

### Task 4: Modify `redistribute_logits()` to accept and use reachable set (torch)

**Files:** `src/tgirl/transport.py`, `tests/test_transport.py`

**Approach:**

Mirror Task 3 for the torch backend. Add `reachable_tokens: frozenset[int] | None = None` parameter to `redistribute_logits()`. When provided:

1. `reachable_idx = torch.tensor(sorted(reachable_tokens), dtype=torch.long)`
2. `valid_mask_R = valid_mask[reachable_idx]`
3. `logits_R = logits[reachable_idx]`
4. Run bypass checks and OT on projected tensors.
5. `invalid_indices_R = torch.where(~valid_mask_R)[0]`
6. `valid_indices_R = torch.where(valid_mask_R)[0]`
7. `embeddings_R = embeddings[reachable_idx]`
8. Cost submatrix from `embeddings_R`.
9. Sinkhorn on smaller cost matrix.
10. Remap: `result = torch.full((vocab_size,), float("-inf"))`, then `result[reachable_idx[valid_indices_R]] = log(combined_probs)`.

When `reachable_tokens=None`, behavior is unchanged.

**Key constraint:** All tensor math in `torch`. No numpy, no MLX.

**Tests (RED first):**

```python
class TestRedistributeLogitsReachableSet:
    def test_reachable_none_unchanged(self):
        """reachable_tokens=None produces same result as before."""

    def test_reachable_set_enables_ot(self):
        """With reachable set, OT runs where it would have been bypassed."""

    def test_reachable_set_result_shape(self):
        """Output shape is (vocab_size,) regardless of reachable set size."""

    def test_unreachable_tokens_negative_inf(self):
        """Tokens not in reachable set have -inf in output."""

    def test_valid_tokens_have_finite_logits(self):
        """Valid tokens within reachable set have finite log-probs."""

    def test_probabilities_sum_to_one(self):
        """exp(output) sums to ~1.0 over finite values."""

    def test_small_reachable_set_runs_ot(self):
        """Reachable set of 100 tokens with 10 valid -> OT runs (1000 < 1M cap)."""

    def test_large_full_vocab_would_bypass(self):
        """Same scenario without reachable set -> OT bypassed (problem_size > 1M)."""

    def test_backward_compatible_signature(self):
        """Existing callers without reachable_tokens still work."""
```

**Validation:** `pytest tests/test_transport.py::TestRedistributeLogitsReachableSet -v`

---

### Task 5: Wire reachable set through sampling loops

**Files:** `src/tgirl/sample.py`, `src/tgirl/sample_mlx.py`, `tests/test_reachable_set.py`

**Approach:**

1. **`run_constrained_generation()`** (`sample.py`): Add `reachable_tokens: frozenset[int] | None = None` parameter. Pass through to `redistribute_logits(..., reachable_tokens=reachable_tokens)`.

2. **`run_constrained_generation_mlx()`** (`sample_mlx.py`): Same addition. Pass through to `redistribute_logits_mlx(..., reachable_tokens=reachable_tokens)`.

3. **`SamplingSession.run()`** (`sample.py`): After `grammar_output = generate_grammar(snapshot)`, extract `grammar_output.reachable_tokens`. Pass to `run_constrained_generation(..., reachable_tokens=grammar_output.reachable_tokens)` and `run_constrained_generation_mlx(..., reachable_tokens=grammar_output.reachable_tokens)`.

4. **Update `generate_grammar()` call**: When a tokenizer_decode and vocab_size are available (they are -- `self._decode` and `self._embeddings.shape[0]`), pass them to `generate()` so it computes the reachable set. This is the activation point.

**Tests (RED first):**

```python
class TestSamplingReachableSetWiring:
    def test_run_constrained_generation_accepts_reachable_tokens(self):
        """run_constrained_generation has reachable_tokens parameter."""

    def test_run_constrained_generation_mlx_accepts_reachable_tokens(self):
        """run_constrained_generation_mlx has reachable_tokens parameter."""

    def test_reachable_tokens_passed_to_transport(self):
        """Transport receives reachable_tokens when grammar provides them."""

    def test_none_reachable_tokens_backward_compatible(self):
        """None reachable_tokens = no change in behavior."""
```

**Validation:** `pytest tests/test_reachable_set.py::TestSamplingReachableSetWiring -v`

---

### Task 6: Benchmark -- measure OT engagement rate before/after

**Files:** `tests/test_reachable_set.py`, `benchmarks/transport_reachable_set.py` (new)

**Approach:**

1. Create a benchmark script that:
   - Constructs a realistic grammar (5 tools with int/str/float parameters)
   - Creates a mock tokenizer_decode with a 250k vocabulary (ASCII-biased: ~10k ASCII tokens, ~240k non-ASCII tokens)
   - Runs `compute_reachable_set()` and reports `|R|` and wall time
   - Simulates constrained generation with varying `n_valid` counts (1, 5, 11, 100, 1000)
   - For each `n_valid`, computes problem_size with and without reachable set
   - Reports OT engagement rate (% of positions where problem_size < max_problem_size)

2. Add integration tests that verify:
   - OT engagement rate improves with reachable set vs without
   - Per-token overhead of reachable-set index projection is <0.5ms
   - `compute_reachable_set()` completes in <2s for 250k vocab

**Tests (RED first):**

```python
class TestReachableSetBenchmark:
    def test_ot_engagement_improves(self):
        """OT engagement rate with reachable set > without reachable set."""

    def test_per_token_overhead_under_threshold(self):
        """Reachable-set projection adds <0.5ms per token."""

    def test_compute_reachable_set_under_2s(self):
        """compute_reachable_set completes in <2s for 250k vocab."""

    def test_reachable_set_size_reasonable(self):
        """Reachable set is >100 and <50000 for a typical 5-tool grammar."""
```

**Validation:** `pytest tests/test_reachable_set.py::TestReachableSetBenchmark -v` and `python benchmarks/transport_reachable_set.py`

---

### Task 7 (Optional): NNMF factorization for large reachable sets

**Files:** `src/tgirl/transport.py`, `src/tgirl/transport_mlx.py`, `tests/test_transport.py`, `tests/test_transport_mlx.py`

**Approach:**

For grammar states where `|R_invalid| x |R_valid|` still exceeds `max_problem_size` after reachable-set pruning (e.g., unconstrained string generation), factorize the cost matrix:

1. `C_R = W @ H` where `W: (|R_invalid|, r)`, `H: (r, |R_valid|)`, `r ~ 32`
2. NNMF preserves non-negativity (costs are cosine distances in [0, 2])
3. Factored Sinkhorn applies the kernel as two sequential operations without materializing the full cost matrix

Pre-compute `W, H` from the reachable-set embeddings at grammar compilation time. Pass to transport as optional parameters.

This task is optional -- the reachable set alone is expected to handle >80% of positions. NNMF targets the remaining edge cases.

**Tests (RED first):**

```python
class TestNNMFFactorization:
    def test_nnmf_produces_valid_factors(self):
        """W and H are non-negative, W @ H approximates C."""

    def test_factored_sinkhorn_matches_full(self):
        """Factored Sinkhorn produces similar transport plan to full Sinkhorn."""

    def test_nnmf_runs_when_reachable_still_too_large(self):
        """NNMF engages when reachable-set cost matrix still exceeds cap."""
```

**Validation:** `pytest tests/test_transport.py::TestNNMFFactorization tests/test_transport_mlx.py::TestNNMFFactorization -v`

## 4. Validation Gates

```bash
# Syntax/Style
ruff check src/tgirl/grammar.py src/tgirl/transport.py src/tgirl/transport_mlx.py src/tgirl/sample.py src/tgirl/sample_mlx.py --fix
mypy src/tgirl/grammar.py src/tgirl/transport.py src/tgirl/transport_mlx.py src/tgirl/sample.py src/tgirl/sample_mlx.py

# Unit Tests (all transport + grammar + new reachable set tests)
pytest tests/test_reachable_set.py tests/test_transport.py tests/test_transport_mlx.py tests/test_grammar.py -v

# Integration Tests
pytest tests/test_integration_transport.py tests/test_integration_grammar.py -v

# Benchmark
python benchmarks/transport_reachable_set.py
```

## 5. Rollback Plan

All changes are backward compatible via `None` defaults:
- `GrammarOutput.reachable_tokens` defaults to `None`
- `redistribute_logits()` / `redistribute_logits_mlx()` ignore `reachable_tokens=None`
- `run_constrained_generation()` / `run_constrained_generation_mlx()` pass `None` when not provided

To disable the feature without reverting code: stop passing `tokenizer_decode` and `vocab_size` to `generate()`. The reachable set will not be computed and all transport behavior reverts to pre-change.

## 6. Uncertainty Log

1. **Regex terminal extraction completeness:** The grammar text parsing approach (regex over Lark EBNF) may miss edge cases in grammar formatting. The Lark parser itself could be used for more robust extraction, but that would add complexity. Starting with regex extraction and hardcoding known common terminals (`ESCAPED_STRING`, `SIGNED_INT`, `SIGNED_FLOAT`, `SYMBOL`) is pragmatic. Flag for review if grammar template format changes.

2. **Tokenizer decode behavior at boundaries:** Some tokenizers produce control characters or empty strings for certain token IDs. The vocab scan should handle these gracefully (skip empty decode results, include control characters that match regex terminals).

3. **Reachable set size for ESCAPED_STRING:** The `ESCAPED_STRING` terminal may match a very large fraction of tokens (any token containing printable ASCII is potentially part of a string). This could limit the effectiveness of the reachable set for string-heavy grammars. Measured data from benchmarks will inform whether NNMF (Task 7) is necessary.

4. **`generate()` signature change:** Adding `tokenizer_decode` and `vocab_size` parameters to `generate()` changes its public API. Both have `None` defaults, so it is backward compatible, but callers that want reachable sets must update their call sites. The `SamplingSession` does this automatically.
