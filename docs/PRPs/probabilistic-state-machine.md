# PRP: Probabilistic State Machine for SamplingSession

## Context

BFCL v4 telemetry (0.8B model, 400 entries) revealed that the current delimiter-based freeform→constrained transition is fundamentally flawed for small/base models:

- **Freeform thinking hurts**: PASS entries average 3.7 freeform tokens, FAIL average 7.2. More thinking = worse.
- **9 entries (2.3%) produce NO_TOOL_CALL** because the model never emits `<tool>`.
- **40 entries (10%) ERROR** — often timeout during freeform rambling.
- **OT barely engages** (4.5% of tokens) because grammar freedom is bimodal: either 1 valid token (forced) or 247k (entire vocab). No middle ground for OT to work in.
- **Confidence is the clearest signal**: PASS logprob -0.03 vs FAIL -0.11 at decision points. The model knows when it's wrong.

The delimiter approach assumes the model understands a protocol (instruct-tuning dependency). tgirl claims model-agnosticism. The fix: make the **orchestrator** decide state transitions using the model's own probability distribution as a signal, not by waiting for the model to emit a specific token sequence.

Additionally, when the model makes a low-confidence choice during constrained generation, we should backtrack and use OT to steer away from the dead-end path.

## Tier

**Standard.** Multi-module architectural refactor touching sample.py, sample_mlx.py, types.py, plus a new state_machine.py module.

## Architecture

### States

```
FREEFORM → ROUTE → CONSTRAINED → EXECUTE → INJECT → FREEFORM
                        ↓                               ↓
                   BACKTRACK → CONSTRAINED             DONE
```

| State | What happens | Exit condition |
|-------|-------------|----------------|
| FREEFORM | Unconstrained generation | TransitionPolicy triggers |
| ROUTE | ToolRouter reranking pass | Routing complete |
| CONSTRAINED | Grammar-constrained generation with confidence monitoring | Grammar accepts, max tokens, or confidence backtrack |
| BACKTRACK | Restore checkpoint, apply OT steering | Steering applied → resume CONSTRAINED |
| EXECUTE | Hy pipeline execution | Pipeline completes |
| INJECT | Result tokens into context | Immediate |
| DONE | Assemble SamplingResult | Terminal |

### Transition Policies (Pluggable)

| Policy | Behavior | Use case |
|--------|----------|----------|
| `delimiter` | Current behavior — wait for `<tool>` | Instruct models, backwards compat |
| `budget(N)` | Force after N freeform tokens | Small models with bounded thinking |
| `immediate` | Skip freeform entirely | Pure tool-calling (BFCL) |
| `confidence` | Markov chain on model signals | General purpose, model-agnostic |
| `composite` | Combine policies (OR/AND) | Delimiter + budget fallback |

### Confidence Signals (Directional)

Confidence works in **opposite directions** for the two generative states:

**FREEFORM → ROUTE**: Triggered by *increasing* confidence toward structured output. The model is converging on a tool call. Signals:
- **Grammar mask overlap** rising: `sum(softmax(logits) * grammar_valid_mask)` — model putting more mass on grammar-valid tokens over successive positions
- **Entropy dropping**: model getting more certain
- **Trend**: sliding window shows increasing overlap / decreasing entropy

**CONSTRAINED → BACKTRACK**: Triggered by *declining* confidence. The model made a bad choice and knows it. Signals:
- **Log prob dropping** below threshold at decision points (high-freedom positions)
- **Trend**: sliding window of recent log probs declining

The Markov chain accumulates evidence over a window. A single `(` in prose barely nudges the freeform→route belief, but `(` + tool-name prefix + dropping entropy over 3 tokens pushes past threshold. Similarly, a single low-confidence token during constrained gen doesn't trigger backtrack, but a sustained decline does.

Weights are tunable and can be fit from telemetry data (we have ground truth transition points from BFCL runs).

### Backtracking with OT Steering

During constrained generation, the confidence monitor watches log probs at high-freedom positions (where the model has real choices, not forced decodes):

1. **Checkpoint** at high-freedom positions (grammar_valid_count > threshold)
2. **Trigger backtrack** when mean log prob over last N tokens declines below threshold
3. **Restore**: create fresh grammar state, replay advance() to checkpoint position
4. **Steer**: `BacktrackSteeringHook` applies strong negative logit_bias on the dead-end token(s) — this happens pre-OT, so OT redistributes mass away from the dead end
5. **Resume** constrained generation from checkpoint

Backtracking cost: O(N) grammar replay + cache recompute. Constrained sequences are 10-50 tokens, so this is tolerable.

### Backtrack Loop Prevention

Simple max-backtracks is insufficient. The real mechanism:

1. **Dead-end accumulation**: Each checkpoint maintains a `dead_end_tokens: frozenset[int]` that grows across retries. First backtrack adds the divergent token. Second backtrack from the same checkpoint adds another. The exclusion set monotonically grows.

2. **Exhaustion detection**: If `len(dead_end_tokens)` exceeds a fraction of the valid tokens at that position (e.g., >50% of grammar-valid tokens excluded), the checkpoint is **sealed** — no more backtracks from it. Accept the best-seen sequence (highest mean log prob across all attempts) and continue forward.

3. **Identity check**: Never backtrack to the same checkpoint with the same exclusion set. Since each backtrack adds at least one new dead-end token, this is guaranteed by construction.

4. **Hard cap**: Global max backtracks per constrained generation pass (default 3) as a safety net. This bounds total compute even if checkpoints are never sealed.

5. **Best-so-far tracking**: Each checkpoint remembers the best token sequence seen across all attempts (by mean log prob at decision points). If all retries are exhausted, the best attempt is used rather than the last one.

## Phased Implementation

### Phase 1: Extract State Machine (no behavioral change)

**New file:** `src/tgirl/state_machine.py` — zero torch/mlx imports, pure control flow

Types:
- `SessionState` enum (FREEFORM, ROUTE, CONSTRAINED, BACKTRACK, EXECUTE, INJECT, DONE)
- `TransitionSignal` (token_position, grammar_mask_overlap, token_entropy, token_log_prob, grammar_freedom, trend_window)
- `TransitionDecision` (should_transition, target_state, reason, confidence)
- `TransitionPolicy` protocol (evaluate method)
- `DelimiterTransitionPolicy` wrapping existing `DelimiterDetector`

**Modify:** `src/tgirl/sample.py`
- `SamplingSession.__init__` gains `transition_policy` param (default: `DelimiterTransitionPolicy`)
- `SamplingSession.run()` refactored to dispatch through state machine
- Identical behavior with `DelimiterTransitionPolicy` — all existing tests pass unchanged

**Test Command:** `pytest tests/ -v`

### Phase 2: Simple Transition Policies + Benchmark

- `BudgetTransitionPolicy(budget=N)`
- `ImmediateTransitionPolicy` (equiv to budget=0)
- Run BFCL with `ImmediateTransitionPolicy` on 0.8B → expect +12% from eliminating NO_TOOL_CALL + ERROR
- Add `--transition-policy` flag to `benchmarks/run_bfcl.py`

**Test Command:** `pytest tests/test_state_machine.py -v`

### Phase 3: Backtracking Infrastructure

**New types in `state_machine.py`:**
- `Checkpoint` (position, tokens_so_far, context_tokens, grammar_text, dead_end_tokens)
- `BacktrackEvent` (checkpoint_position, trigger_position, trigger_log_prob, dead_end_tokens_added)
- `ConstrainedConfidenceMonitor` (log_prob_threshold, window_size, freedom_threshold, max_backtracks)

**New hooks:**
- `BacktrackSteeringHook` in `sample.py` (torch)
- `BacktrackSteeringHookMlx` in `sample_mlx.py`

**Modify** `run_constrained_generation` / `run_constrained_generation_mlx`:
- Accept optional `confidence_monitor` and `steering_hooks`
- Checkpoint at high-freedom positions
- On backtrack trigger: return partial result with backtrack signal
- State machine handles replay and retry

**Extend** `TelemetryRecord` in `types.py`:
- `backtrack_events: list[BacktrackEvent] = []`
- `state_transitions: list[tuple[str, str, float]] = []`

**Test Command:** `pytest tests/test_state_machine.py -v -k backtrack`

### Phase 4: Confidence Markov Chain

- `ConfidenceTransitionPolicy` with tunable weights (w_readiness, w_certainty, w_quality, w_trend)
- `compute_transition_signal` helper (framework-agnostic via callable softmax/sum/log)
- `CompositeTransitionPolicy` with OR/AND modes
- Benchmark with confidence policy on 0.8B and 9B models
- Tune weights from telemetry data

**Test Command:** `pytest tests/test_state_machine.py -v -k confidence`

## Critical Files

| File | Change |
|------|--------|
| `src/tgirl/state_machine.py` | **New.** State enum, transition policies, checkpoint types, signal computation, confidence monitor |
| `src/tgirl/sample.py` | Refactor `SamplingSession.run()` to state machine dispatch; add `transition_policy` param; extend constrained gen with checkpointing; `BacktrackSteeringHook` |
| `src/tgirl/sample_mlx.py` | Mirror constrained gen changes; `BacktrackSteeringHookMlx` |
| `src/tgirl/types.py` | Extend `TelemetryRecord` with backtrack/transition fields |
| `benchmarks/run_bfcl.py` | Add `--transition-policy` flag |
| `benchmarks/analyze_telemetry.py` | New analyses for backtrack frequency, state transitions |

## Architectural Constraints

- `state_machine.py` has **zero torch/mlx imports** — pure Python + Pydantic
- OT transport remains **zero-coupled** — steering via pre-OT logit_bias (existing hook system), not cost matrix modification
- `DelimiterTransitionPolicy` is default — **backwards compatible**, all existing tests pass
- Both torch and MLX backends supported throughout

## Verification

```bash
# Phase 1: all existing tests pass, no behavioral change
pytest tests/ -v

# Phase 2: BFCL with immediate policy
python benchmarks/run_bfcl.py --model mlx-community/Qwen3.5-0.8B-MLX-4bit \
    --category simple_python --model-name tgirl-0.8b-immediate \
    --transition-policy immediate

# Phase 3: backtracking test
pytest tests/test_state_machine.py -v -k backtrack

# Phase 4: confidence policy benchmark
python benchmarks/run_bfcl.py --model mlx-community/Qwen3.5-0.8B-MLX-4bit \
    --category simple_python --model-name tgirl-0.8b-confidence \
    --transition-policy confidence
```
