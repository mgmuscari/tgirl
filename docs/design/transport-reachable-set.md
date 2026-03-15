# Transport Reachable Set Optimization — Design Document

**Status:** Design
**Date:** 2026-03-15
**Architecture block:** STB (Semantic Transport Bus) + GSU (Grammar Selection Unit)
**Depends on:** None (independent of ADSR modulation matrix)

---

## 1. Problem

OT (Sinkhorn optimal transport) is bypassed on ~80% of constrained generation tokens due to `max_problem_size` (1M elements). The cost matrix is `n_invalid × n_valid`, where `n_invalid ≈ 248k` (full vocab minus valid tokens). For a typical grammar position with 11 valid tokens (digits during integer generation), the cost matrix is `248k × 11 = 2.7M` — over the 1M cap.

When OT is bypassed, the STB is offline. The modulation matrix can set `transport_epsilon` to any value, but it has no effect. The model's logit distribution is grammar-masked but not semantically transported.

## 2. Observation

Of the 248k tokens in a typical vocabulary, ~95% are provably unreachable from any grammar state: Chinese characters, Korean, emoji, code tokens, markdown formatting. The grammar's terminal vocabulary defines the ceiling — tokens that don't match any terminal can never be valid.

## 3. Solution

**Pre-compute the reachable token set at grammar compilation time.** Restrict OT to operate within this set only.

### Reachable set extraction

Parse the Lark grammar text. For each terminal:

- **String literals** (`"("`, `"tool_name"`, `" "`): tokenize, collect all token IDs that the tokenizer produces for these strings
- **Regex terminals** (`SIGNED_INT: /[+-]?[0-9]{1,18}/`, `SIGNED_FLOAT`, `ESCAPED_STRING`): for each token ID in vocab, check if `tokenizer.decode([tid])` matches the regex. Collect matching IDs.
- **Imported terminals** (`%import common.ESCAPED_STRING`): resolve the Lark common grammar definition, extract the regex, scan as above.

Union all collected token IDs = **reachable set `R`**.

Expected size: `|R| ≈ 5,000–10,000` for a typical tool grammar (digits, ASCII letters, quotes, parens, spaces, tool-name subwords).

### Computation

```python
def compute_reachable_set(
    grammar_text: str,
    tokenizer_decode: Callable[[list[int]], str],
    vocab_size: int,
) -> frozenset[int]:
    """Extract the set of token IDs reachable from any grammar state.

    Parses grammar terminals, expands regex patterns, and scans
    the vocabulary for matching tokens.
    """
```

Pre-compute once per grammar compilation. Cache alongside the grammar output.

### OT with reachable set

```python
# Before (per token):
# cost_matrix shape: (248k - n_valid, n_valid) → often >1M → bypassed

# After (per token):
valid_in_R = valid_mask[reachable_set]     # (|R|,)
invalid_in_R = ~valid_in_R                  # (|R|,)
# cost_matrix shape: (|R_invalid|, |R_valid|) → typically <100k → runs

# Mass on unreachable tokens is zeroed before sampling
```

### Expected impact

| Metric | Current | With reachable set |
|--------|---------|-------------------|
| Cost matrix size (integer gen, n_valid=11) | 2.7M (bypassed) | 88k (runs) |
| Cost matrix size (string gen, n_valid=~5k) | 1.2B (bypassed) | 25M (may still bypass) |
| OT engagement rate | ~20% of tokens | ~80%+ of tokens |
| Pre-computation cost | 0 | ~1s (one-time vocab scan) |

### Optional: NNMF factorization

For grammar states where `|R_invalid| × |R_valid|` still exceeds the cap (e.g., string generation with many valid tokens), factorize the reachable-set cost matrix:

```python
C_R ≈ W @ H    # W: (|R_invalid|, r), H: (r, |R_valid|), r ≈ 32
```

NNMF preserves non-negativity (costs are cosine distances ∈ [0, 2]), which maintains correctness of the log-domain Sinkhorn kernel. The factored Sinkhorn applies the kernel as two sequential operations without materializing the full cost matrix.

**Pre-compute at grammar compilation time:**
1. `reachable_embeddings = embeddings[reachable_set]`
2. `W, H = NNMF(reachable_embeddings_cost_matrix, rank=r)`
3. Cache `W`, `H` alongside the grammar

**Per token:** Sinkhorn uses `W @ H` instead of `C`. Memory: `|R|×r + r×n_valid` instead of `|R_invalid|×n_valid`.

## 4. Implementation sequence

1. **`compute_reachable_set()`** — grammar terminal extraction + vocab scan. New function in `grammar.py` or a new `transport_cache.py`.
2. **Cache reachable set on `GrammarOutput`** — add `reachable_tokens: frozenset[int]` field.
3. **Pass reachable set to `redistribute_logits_mlx`** — new parameter. When provided, restrict cost matrix computation to reachable tokens only.
4. **Remap valid mask** — project full-vocab valid mask to reachable-set indices before OT, remap back after.
5. **Benchmark** — BFCL before/after, measure OT engagement rate and accuracy impact.
6. **Optional: NNMF factorization** — for grammar states that still exceed cap after reachable-set pruning.

## 5. Architectural fit

This is a GSU → STB interface enhancement. The GSU (grammar compilation) produces the reachable set as a byproduct of grammar analysis. The STB (transport) uses it to restrict its operating domain. No other blocks are affected.

The reachable set is a static property of the grammar, not a per-token signal. It's computed once and cached — not part of the ADSR modulation matrix's per-token signal chain.
