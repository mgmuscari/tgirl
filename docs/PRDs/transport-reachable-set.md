# PRD: Transport Reachable Set Optimization

## Status: DRAFT
## Author: agent (Proposer)
## Date: 2026-03-15
## Branch: feature/transport-reachable-set

## 1. Problem Statement

Optimal transport (OT) redistribution via Sinkhorn is bypassed on approximately 80% of constrained generation tokens due to the `max_problem_size` cap (1M elements). The cost matrix has shape `(n_invalid, n_valid)`, where `n_invalid` is typically ~248k (full vocabulary minus valid tokens). For a grammar position with 11 valid tokens (e.g., digit generation), the cost matrix is 248k x 11 = 2.7M elements -- well over the 1M cap. When OT is bypassed, the Semantic Transport Bus (STB) is offline: probability mass from invalid tokens is discarded rather than redistributed to semantically similar valid tokens. The model's intent is lost.

This affects every user of the library. The transport layer is designed to preserve the model's distributional intent during grammar-constrained generation, but it cannot function when the problem size exceeds the cap. The ADSR modulation matrix can set `transport_epsilon` to any value, but it has no effect on a bypassed STB.

## 2. Proposed Solution

Pre-compute the **reachable token set** at grammar compilation time. Of the ~248k tokens in a typical vocabulary, approximately 95% are provably unreachable from any grammar state: CJK characters, emoji, code tokens, markdown formatting tokens, etc. The grammar's terminal vocabulary defines the ceiling -- tokens that cannot match any terminal can never be valid at any grammar position.

By restricting OT to operate within the reachable set (expected size ~5,000-10,000 tokens) instead of the full vocabulary, the cost matrix shrinks from `(248k, n_valid)` to `(|R_invalid|, n_valid)`, where `|R_invalid| <= |R| <= ~10k`. For the integer generation case: from 2.7M elements (bypassed) to ~88k elements (runs). This brings most grammar positions under the 1M cap.

The reachable set is a static property of the grammar, computed once per grammar compilation and cached on `GrammarOutput`. It is not a per-token signal -- it does not interact with the ADSR modulation matrix's per-token signal chain.

An optional NNMF factorization path handles the remaining edge cases where `|R_invalid| x |R_valid|` still exceeds the cap (e.g., unconstrained string generation with ~5k valid tokens).

## 3. Architecture Impact

### Files/modules affected

- **`src/tgirl/grammar.py`** -- Add `compute_reachable_set()` function for grammar terminal extraction + vocab scan. Add `reachable_tokens: frozenset[int] | None` field to `GrammarOutput`.
- **`src/tgirl/transport.py`** -- Modify `redistribute_logits()` to accept optional `reachable_tokens` parameter. When provided, project `valid_mask` and `logits` to reachable-set indices before OT, remap back after.
- **`src/tgirl/transport_mlx.py`** -- Same modification for `redistribute_logits_mlx()`.
- **`src/tgirl/sample.py`** -- Wire `grammar_output.reachable_tokens` through `run_constrained_generation()` to `redistribute_logits()`.
- **`src/tgirl/sample_mlx.py`** -- Same wiring for MLX path via `redistribute_logits_mlx()`.

### Data model changes

- `GrammarOutput` gains `reachable_tokens: frozenset[int] | None = None` field (backward compatible, defaults to None).
- `redistribute_logits()` and `redistribute_logits_mlx()` gain `reachable_tokens: frozenset[int] | None = None` parameter (backward compatible).
- `run_constrained_generation()` and `run_constrained_generation_mlx()` gain `reachable_tokens: frozenset[int] | None = None` parameter (backward compatible).

### API changes

All API changes are additive with None defaults -- fully backward compatible. Existing callers see no behavior change until `reachable_tokens` is explicitly passed.

### Dependency additions

None. Uses only `re` (stdlib) for regex terminal extraction and existing `lark` grammar parsing. No new external dependencies.

## 4. Acceptance Criteria

1. `compute_reachable_set()` correctly extracts all token IDs reachable from grammar terminals: string literals are tokenized and all constituent token IDs collected; regex terminals (`SIGNED_INT`, `SIGNED_FLOAT`, `ESCAPED_STRING`) are expanded by scanning the vocabulary; imported terminals (`%import common.ESCAPED_STRING`) are resolved to their regex definitions.
2. `GrammarOutput.reachable_tokens` is populated when `generate()` is called, containing a `frozenset[int]` of reachable token IDs.
3. Same registry snapshot produces same reachable set (determinism).
4. OT engagement rate increases from ~20% to >80% of constrained generation tokens on standard BFCL benchmark scenarios with integer and tool-name arguments.
5. No accuracy regression on BFCL benchmarks -- generated tool calls remain correct at the same or better rate.
6. Per-token overhead of reachable-set masking (projecting valid_mask to reachable indices + remapping back) is <0.5ms on both torch and MLX backends.
7. `redistribute_logits()` and `redistribute_logits_mlx()` produce identical results to their pre-change behavior when `reachable_tokens=None` (backward compatibility).
8. All tensor math uses native framework operations: torch ops for `transport.py`, MLX ops for `transport_mlx.py`. No cross-framework conversions. No Python list comprehensions on tensor data.
9. Transport module zero-coupling invariant is preserved: `transport.py` and `transport_mlx.py` have no imports from `tgirl.grammar` or `tgirl.registry`. The reachable set is passed in as a plain `frozenset[int]`.
10. One-time `compute_reachable_set()` cost is <2s for a vocabulary of 250k tokens with a typical 5-tool grammar.

## 5. Risk Assessment

- **Reachable set too large:** If a grammar uses very permissive terminals (e.g., unconstrained `ESCAPED_STRING` matching most vocabulary tokens), the reachable set may not shrink the problem enough. The NNMF factorization path (optional Task 7) addresses this. Mitigation: the reachable set is always a subset of the full vocabulary, so it is never worse than the current behavior.
- **Regex terminal scanning is O(vocab_size):** Scanning 250k tokens against regex patterns takes ~1s. This is a one-time cost at grammar compilation, not per-token. Acceptable.
- **Token decode inconsistency:** Some tokenizers produce different strings for `decode([tid])` vs the token's actual byte representation. Mitigation: use the tokenizer's own decode function (passed as a callable), same approach used by `NestingDepthHookMlx`.
- **Grammar format changes:** If the Lark grammar template format changes, the terminal extraction logic must be updated. Mitigation: the extraction parses the grammar text, which is already a stable output format.

## 6. Open Questions

1. Should `compute_reachable_set()` live in `grammar.py` or a new `transport_cache.py`? The design doc suggests either. Recommendation: `grammar.py`, since it operates on grammar text and is called during grammar generation. The result is a `frozenset[int]` with no transport coupling.
2. Should the reachable set be recomputed when the tokenizer changes, or cached across tokenizer instances? Recommendation: tie to both grammar text hash and tokenizer identity. For now, compute per `generate()` call -- caching is a follow-up optimization.
3. What is the right behavior when `reachable_tokens` is provided but the cost matrix still exceeds `max_problem_size`? Recommendation: fall back to standard masking (bypass), same as current behavior. Log the reduced problem size for observability.

## 7. Out of Scope

- **Raising `max_problem_size`:** The reachable set optimization reduces problem size at the source; it does not change the OT algorithm's memory characteristics.
- **ADSR modulation matrix integration:** The reachable set is a static grammar property, not a per-token dynamic signal. It does not interact with the ADSR modulation layer.
- **Vocabulary-aware grammar generation:** Generating grammars that are aware of the tokenizer's vocabulary (e.g., constraining string terminals to only produce tokens the model has seen) is a separate optimization.
- **Multi-grammar reachable set merging:** When multiple grammars are used in sequence (e.g., routing grammar + execution grammar), merging their reachable sets is not in scope.
- **Persistent caching of reachable sets across sessions:** The reachable set is computed per grammar generation call. Disk caching is a follow-up.
