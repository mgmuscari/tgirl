# Vocabulary Partitioning — Content vs Control Token Masking

**Status:** Design
**Date:** 2026-03-15
**Architecture block:** GSU (Grammar Selection Unit)
**Related:** cycle-gate.md, adsr-envelope.md, transport-reachable-set.md

---

## Problem

Models trained with chat/tool-calling templates have tokens that serve dual roles: `</tool>` is both a string of characters and a control signal meaning "end tool call." Inside `ESCAPED_STRING`, the grammar says `</tool>` is valid content. But the model's weights treat it as a control signal — emitting it triggers trained behaviors (reasoning, mode switching, repetition) that derail string generation.

BFCL telemetry shows 5 error entries where the model enters a string argument, then generates `</tool>`, `<tool>`, `<|im_end|>`, or similar control tokens as "escape hatches" — followed by freeform reasoning that burns 512 tokens.

## Insight

The model has two vocabularies in the same token space:

- **Content vocabulary** — words, numbers, punctuation. These represent meaning.
- **Control vocabulary** — delimiters, EOS, role markers, special tokens. These represent structure.

The grammar treats them as one vocabulary. The engine should separate them: **during content-position generation (string arguments, numbers), mask control tokens from the valid set.** The model can spell any control sequence character-by-character if it truly needs the literal text, but it cannot emit the single-token shortcut that triggers trained control behavior.

## Mechanism

### Control token identification

Extract from tokenizer metadata at model load time:

```python
def extract_control_tokens(tokenizer) -> frozenset[int]:
    """Identify all tokens the model treats as structural control."""
    control = set()

    # All added special tokens (chat template markers)
    for token_str, token_id in tokenizer.added_tokens_encoder.items():
        control.add(token_id)

    # EOS, BOS, pad
    for attr in ['eos_token_id', 'bos_token_id', 'pad_token_id']:
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            control.add(tid)

    return frozenset(control)
```

For Qwen3.5, this produces ~30 token IDs including `<tool_call>`, `</tool_call>`, `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, etc.

### Grammar position classification

The grammar state knows whether the current parse position is structural or content. Two approaches:

**Approach A (heuristic):** If the valid mask includes the quote-close token (`"`), we're inside a string. Apply control token masking.

**Approach B (grammar introspection):** Extend `GrammarState` protocol with `active_terminal() -> str | None`. When the active terminal is `ESCAPED_STRING`, `SIGNED_INT`, or `SIGNED_FLOAT`, apply control token masking. Requires llguidance to expose terminal tracking.

**Approach C (precomputed):** At grammar compile time, walk the grammar automaton and classify each state as "structural" or "content" based on which terminal it's expanding. Store the classification alongside the grammar state. No runtime detection needed.

Approach A is sufficient for v1. Approach C is the architecturally clean solution.

### Application

In the constrained generation loop, after grammar mask and stop token masking:

```python
# Mask control tokens during content positions
if control_token_ids and _is_content_position(valid_mask, quote_close_token_id):
    control_mask = mx.ones(valid_mask.shape, dtype=mx.bool_)
    indices = mx.array(list(control_token_ids))
    control_mask[indices] = False
    valid_mask = valid_mask & control_mask
```

Same pattern as stop token masking and cycle gate — mask-level intervention before OT.

### Effect

The model cannot emit `</tool>` (token 248059) inside a string. It can emit `<` (token 27), `/` (token 14), `t` (token 83), `o` (token 78), `o` (token 78), `l` (token 75), `>` (token 29) — seven tokens that spell the same characters without triggering control behavior. The 7-token path is unlikely to be generated because the model has no training incentive to spell out `</tool>` character by character inside a string argument.

## Interaction with other systems

- **Stop token masking:** Already masks EOS during non-accepting grammar states. Vocabulary partitioning extends this principle to all control tokens during content positions.
- **Cycle gate:** Operates on content tokens (repeated values). Vocabulary partitioning prevents control tokens from appearing in the cycle in the first place.
- **ADSR envelope:** Operates on the logit distribution. Vocabulary partitioning operates on the valid mask — upstream of the envelope.
- **Transport reachable set:** Control tokens are included in the reachable set (they're valid at structural positions) but masked at content positions. The reachable set is position-independent; vocabulary partitioning is position-dependent.

## Implementation

1. `extract_control_tokens(tokenizer) -> frozenset[int]` — utility in sample.py
2. `_is_content_position(valid_mask, quote_token_id) -> bool` — heuristic: True if quote-close is in valid set
3. Apply masking in constrained gen loop (sample_mlx.py and sample.py)
4. Pass `control_token_ids` through `SamplingSession` (same pattern as `stop_token_ids`)
5. Test: verify control tokens are masked during string generation, allowed during structural positions
