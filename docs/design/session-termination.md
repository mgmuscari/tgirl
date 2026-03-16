# Session Termination — SCU should decide when to stop

**Status:** Bug / Design
**Date:** 2026-03-15
**Architecture block:** SCU (Stochastic Control Unit)
**Observed in:** BFCL 9B benchmark — model calls same tool 3-6x per entry

---

## Problem

After a tool call executes and results are injected, the SCU transitions back to FREEFORM. The model generates more freeform tokens, the transition policy fires again, and the model makes another tool call — often identical to the first. This repeats until `max_tool_cycles` is exhausted.

The model has no mechanism to decide "I have enough information, I should stop." The SCU's transition policies detect when to *start* a tool call but not when to *end the session*. The result: 6 identical calls to `(calculate_triangle_area 10 5 "units")` at 22 seconds when one call at ~4 seconds was sufficient.

## Root Cause

The session loop structure is:

```
for _ in range(max_tool_cycles + 1):
    FREEFORM → detect delimiter → CONSTRAINED → EXECUTE → INJECT → back to FREEFORM
```

After INJECT, the model returns to FREEFORM with the tool result in context. A well-aligned instruct model would generate a natural language response and stop. But:

1. The transition policy fires again on the freeform output (it detects tool-calling intent in the model's reasoning)
2. The model has no "I'm done" signal that the SCU recognizes
3. `max_tool_cycles` is a hard cap, not an intelligent termination condition

## Desired Behavior

After tool execution and result injection, the SCU should evaluate whether the session should continue or terminate. Signals:

- **Result sufficiency**: the tool returned a valid result (not an error). For simple tool calls, one successful result is usually enough.
- **Duplicate detection**: the model is about to call the same tool with the same arguments. Suppress the transition.
- **Model confidence**: if the freeform output after result injection has high confidence and low entropy, the model is generating a summary/response, not preparing another tool call.
- **Explicit termination**: the system prompt could instruct the model to emit a termination signal (e.g., a specific token or phrase) when it has enough information.

## Proposed Solutions

### Short-term: Duplicate call suppression

Track previous tool calls (pipeline source + arguments). If the constrained gen produces an identical call, skip execution and return to freeform. This is cheap and catches the exact pattern observed in BFCL.

### Medium-term: Result-aware transition policy

After INJECT, the transition policy should factor in whether a successful result was just injected. A new policy (or extension to existing policies) that requires stronger transition signals after a successful tool call — the bar for "start another tool call" should be higher than the bar for "start the first tool call."

### Long-term: CSG-driven session termination

The CSG monitors the freeform output after result injection. If the model's entropy is low and confidence is high for N consecutive tokens, the session is in "summarization mode" — the model is presenting results, not reasoning about what to do next. The SCU can suppress further transitions.

This is a natural extension of the ADSR concept: the session itself has phases (query → reason → call → present), and the SCU should recognize which phase it's in.

## Workaround

Set `max_tool_cycles=1` for benchmarks where each entry requires exactly one tool call (BFCL simple_python). Not a fix — multi-step workflows legitimately need multiple cycles.
