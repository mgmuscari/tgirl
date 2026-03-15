# Inference Safety Hooks — Design Document

**Date:** 2026-03-14
**Status:** Implemented
**Modules:** `sample.py`, `sample_mlx.py`, `grammar.py`, `instructions.py`, `rerank.py`, `types.py`

---

## Problem

BFCL v4 benchmarking on Qwen3.5-0.8B revealed multiple classes of inference failure that are addressable through infrastructure, not model capability:

1. **Degenerate token loops** — Model repeats `(-> (-> (-> ...` for 128 tokens, producing unparseable output
2. **Premature EOS** — Model emits `<|im_end|>` during constrained generation instead of tool calls
3. **Grammar dead ends** — llguidance produces zero valid tokens, fallback samples garbage (token 0 = `!`)
4. **Missing parameter context** — Model receives type signatures but no parameter descriptions, generating wrong argument values
5. **Single-tool routing** — Router restricts grammar to one tool, eliminating composition
6. **8.3s per-token bottleneck** — Python list comprehension over 248k-element MLX tensor

## Solutions

### 1. Composition-Aware Routing (grammar.py, instructions.py, rerank.py)

**Problem:** Router grammar accepted exactly one tool name. Wrong routing = wrong grammar = failed generation.

**Solution:**
- `generate_routing_grammar(snapshot, top_k)` now produces a space-separated list grammar using trailing optional chains (compatible with both Lark and llguidance)
- `generate_routing_prompt()` instructs model to list tools needed for composition
- `ToolRouter.route()` parses multi-tool output, deduplicates, respects `top_k`
- `SamplingSession` filters snapshot to selected tools → multi-tool grammar → composition possible

**Backwards compat:** `top_k=1` (default) produces identical single-choice behavior.

### 2. Lowercase Lark Rule Names (grammar.py)

**Problem:** Tool names like `calculate_BMI` produced rule names `call_calculate_BMI`. Lark treats uppercase names as terminal references, causing llguidance to reject the grammar silently. Result: zero valid tokens at every position → 128 tokens of garbage.

**Solution:** All generated rule names are lowercased (`.lower()`). Tool names in string literals (what the model outputs) are unchanged — only internal Lark rule names are affected.

### 3. Stop Token Masking (sample.py, sample_mlx.py)

**Problem:** Model emits `<|im_end|>` repeatedly during constrained generation. The grammar guide marks EOS as valid, so it gets sampled.

**Solution:** When `stop_token_ids` is provided and the grammar is **not in an accepting state**, those token IDs are zeroed out of the valid mask. Once the grammar accepts (complete expression), stop tokens are allowed — the model can naturally terminate after valid output. Freeform generation is unaffected.

**Implementation detail:** MLX valid masks are `dtype=bool`. The masking uses `mx.logical_and` with a bool stop mask to avoid dtype conflicts with downstream numpy operations in `transport_mlx.py`.

### 4. Grammar Dead-End Abort (sample.py, sample_mlx.py)

**Problem:** When `valid_count == 0`, the fallback path samples uniformly over zero valid tokens, producing `token_id=0` (typically `!` or similar) for every remaining position.

**Solution:** If `valid_count == 0` after grammar mask + stop token masking, the generation loop breaks immediately with a warning log. The partial output is returned rather than 128 tokens of garbage.

**Root cause note:** `valid_count == 0` should never occur in a well-formed grammar. It indicates either a grammar compilation failure (e.g., uppercase rule names) or a tokenizer-grammar mismatch. The abort is a safety net, not a fix.

### 5. Repetition Penalty Hook with Cycle Detection (sample.py, sample_mlx.py)

**Problem:** Multi-token degenerate patterns like `(-> (-> (-> ...` bypass per-token repetition penalties because each individual token doesn't exceed the threshold in the window.

**Solution:** Two-layer penalty:

1. **Window-based:** Counts each token's occurrences in the last `window` positions. Tokens exceeding `max_repeats` receive an escalating bias: `bias * (count - max_repeats)`.

2. **Cycle detection:** Checks if the token suffix contains a repeating cycle by comparing `tokens[-k:] == tokens[-2k:-k]` for k from 1 to `max_period`. If a cycle is detected, all tokens in the cycle receive `cycle_bias` on top of any window penalty.

**Algorithm:** The cycle detection is O(k * max_period) per token, where k is bounded by `max_period` (default 16). This is the suffix-based cycle detection pattern — same core idea as Floyd's tortoise and hare but adapted for discrete sequences with known maximum period.

### 6. Nesting Depth Hook (sample.py, sample_mlx.py)

**Problem:** The model can open arbitrarily deep s-expression nesting, then hit `constrained_max_tokens` before closing all parens, producing invalid Hy.

**Solution:** The hook pre-computes a paren delta for every token in the vocabulary at init time (one-time O(vocab_size) scan). During generation, it tracks nesting depth incrementally via `advance(token_id)`. When `remaining_tokens <= depth + margin`, all tokens with net-positive paren delta receive `-100` bias, forcing the model to close expressions rather than open new ones.

**Generalizability:** The hook is parameterized by `open_char` and `close_char` (default `(` and `)`). It works with any tokenizer — the vocab scan at init handles multi-character tokens like `("` or ` (` that contain opening parens.

**Statefulness:** The hook has mutable `_depth` state. The constrained generation loop calls `hook.reset()` at the start of each pass and `hook.advance(token_id)` after each sampled token.

### 7. Parameter Descriptions in System Prompt (types.py, registry.py, instructions.py)

**Problem:** `ParameterDef` had no `description` field. BFCL schemas contain parameter descriptions like `"Preferred unit of distance (optional, default is 'km')"` but this information was dropped during registration. The model saw `<unit:str>` and had to guess.

**Solution:**
- `ParameterDef` gains `description: str = ""` field
- `register_from_schema()` passes `prop_schema.get("description", "")` through
- `generate_tool_doc()` emits parameter descriptions below the tool signature

### 8. Native Transition Signal Computation (sample.py, sample_mlx.py)

**Problem:** The "framework-agnostic" `compute_transition_signal()` in `state_machine.py` required converting 248k-element tensors to Python lists. `[float(v) for v in mx_array]` on 248k elements takes **8.3 seconds**; `.tolist()` takes 6ms.

**Solution:** Replaced with native variants:
- `compute_transition_signal_torch()` in `sample.py` — uses `torch.softmax`, `torch.log`, `torch.sum`, `torch.dot`
- `compute_transition_signal_mlx()` in `sample_mlx.py` — uses `mx.softmax`, `mx.log`, `mx.sum`

The original `compute_transition_signal()` remains in `state_machine.py` for test use with small Python lists. Production code never calls it.

**Architectural principle:** `state_machine.py` owns control logic (policies, signal types, decisions). Sampling modules own computational primitives that produce those signals.

## Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| Per-entry latency | 35s | 3-5s |
| Per-token throughput | 8.3s (bottleneck) | 6.5ms |
| Dead-end errors | 15-25 per run | 0 |
| EOS spam errors | 16 per run | 0 |
| AST accuracy (0.8B, simple_python) | 44% | ~50% (model-limited) |

## Hook Execution Order

During constrained generation, hooks execute in registration order:

```
1. GrammarTemperatureHook — adjusts temperature based on grammar freedom
2. RepetitionPenaltyHook — window + cycle detection penalties
3. NestingDepthHook — prevents unclosable expressions
```

All hooks return `ModelIntervention` objects. Interventions are merged (last-writer-wins per field, logit_bias values are summed). The merged intervention is applied before OT redistribution.

After token sampling:
```
1. hook.advance(token_id) — called on any hook with an advance method
2. grammar_state.advance(token_id) — updates grammar parse state
```

## Configuration

```python
SessionConfig(
    constrained_max_tokens=128,  # Budget for s-expression generation
    max_tool_cycles=10,          # Max freeform→constrained round-trips
)

RerankConfig(
    top_k=len(tools),  # Allow router to return all relevant tools
)

# Hooks
GrammarTemperatureHook(base_temperature=0.5)
RepetitionPenaltyHook(window=8, max_repeats=2, bias=-20.0, cycle_bias=-50.0)
NestingDepthHook(
    max_tokens=128,
    tokenizer_decode=tokenizer.decode,
    vocab_size=vocab_size,
    margin=2,
)

# Stop tokens
stop_token_ids = [tokenizer.eos_token_id, ...]  # Masked during constrained gen
```
