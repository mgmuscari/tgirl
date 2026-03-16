# Cycle Gate — Gated Reverb for Token Repetition

**Status:** Design
**Date:** 2026-03-15
**Blocked by:** ADSR matrix tuning session
**Related:** ADSR modulation matrix, transport reachable set

---

## Problem

The ADSR modulation matrix's cycle detection identifies degenerate token repetition (e.g., `(-> (-> (-> ...` repeating with period 3) but can only respond with logit bias — making tokens less likely, not impossible. When the grammar only offers 3 valid tokens and the cycle tokens are among them, a -30 logit bias isn't enough. The model continues cycling.

Additionally, attack phase temperature (0.80) is higher than the pre-ADSR static hook (0.30), which *amplifies* cycling by encouraging exploration in a grammar state where the only "exploration" available is more nesting.

## Insight: Gated Reverb

In audio production, gated reverb lets a signal ring naturally but slams a hard gate when repetition crosses a threshold. The reverb doesn't fade out — it *stops*. The result sounds intentional, not reluctant.

Applied to inference: token repetition is the reverb tail. The gate detects the cycle and removes the cycling tokens from the valid mask entirely — not penalized, *removed*. Zero probability, not low probability. The model is forced to find a different valid continuation.

## Mechanism

**Progressive gating:**

1. **First occurrence:** Natural. Gate open.
2. **Second occurrence:** Gate open. CSG notes the pattern.
3. **Third occurrence (K=3):** Gate closes. Cycling tokens removed from `valid_mask` before OT and sampling. Hard cutoff.

**Implementation:** Mask-level intervention, same mechanism as stop token masking and NestingDepthHook opener penalty. Applied after grammar mask, before hooks:

```python
# In constrained generation loop, after valid_mask computation:
if cycle_gate.is_triggered(token_history):
    cycle_tokens = cycle_gate.get_cycle_token_ids()
    for tid in cycle_tokens:
        valid_mask[tid] = False  # hard gate, not logit bias
```

**Parameters (configurable via mod matrix or standalone):**
- `hold_count`: How many cycle repetitions before the gate closes (default: 3)
- `gate_duration`: How many tokens the gate stays closed (default: until grammar state changes significantly — freedom shifts >50%)
- `min_period`: Minimum cycle length to detect (default: 1)
- `max_period`: Maximum cycle length to detect (default: 16)

## Why mask, not bias

| Approach | Effect | Behavior |
|----------|--------|----------|
| Logit bias (-30) | Low probability | Model reluctantly generates something else, may still cycle if bias < logit magnitude |
| Logit bias (-100) | Very low probability | Effectively zero but still wastes OT computation on near-zero mass |
| **Mask removal** | **Zero probability** | **Token impossible. Model decisively moves on. OT doesn't waste compute.** |

The mask removal is structurally cleaner and computationally cheaper. It's the same mechanism the system already uses for stop tokens and grammar constraints.

## Interaction with ADSR

The cycle gate operates at the mask level (before hooks). The ADSR modulation matrix operates at the logit level (via hooks). They're complementary:

- ADSR shapes parameters smoothly across phases
- Cycle gate provides hard structural intervention when the oscillator locks into a degenerate mode

In synthesis terms: ADSR is the envelope generator. The cycle gate is the noise gate on the effects return. Different signal chain position, different purpose.

## Also needed: ADSR default matrix fixes

The cycling problem also requires matrix weight adjustments:

1. **Attack temperature too high** (0.80) — should be closer to 0.50 for the 0.8B model
2. **Cycle gate → temperature routing** — when cycle detected, temperature should drop (not just repetition bias)
3. **Opener penalty too gentle** (-20 vs old -100) — depth normalization at `range_max=10` caps the effective penalty

These are tuning issues, not architectural issues. Address during the ADSR matrix tuning session using 0.8B BFCL telemetry.
