# PRP: Probabilistic State Machine — Deferred Items

Follow-up from the initial PSM implementation (docs/reviews/code/probabilistic-state-machine-review.md).

## Tasks

### Task 1: `compute_transition_signal` helper

Framework-agnostic helper that computes TransitionSignal from raw logits and grammar mask.

**Location:** `src/tgirl/state_machine.py` (zero torch/mlx imports — accept callables)

```python
def compute_transition_signal(
    token_position: int,
    logits: Any,  # framework tensor
    grammar_valid_mask: Any,  # framework tensor
    softmax_fn: Callable,  # framework softmax
    sum_fn: Callable,  # framework sum
    log_fn: Callable,  # framework log
    vocab_size: int,
) -> TransitionSignal:
```

Computes:
- `grammar_mask_overlap`: `sum(softmax(logits) * grammar_valid_mask)` — probability mass on grammar-valid tokens
- `token_entropy`: `-sum(p * log(p))` over the distribution
- `token_log_prob`: log prob of the sampled token (passed in or computed)
- `grammar_freedom`: `sum(grammar_valid_mask) / vocab_size`

**Wire into** `SamplingSession.run()` freeform loop in `src/tgirl/sample.py` — replace the hardcoded 0.0 fields.

**Test Command:** `pytest tests/test_state_machine.py -v -k compute_transition_signal`

### Task 2: `BacktrackSteeringHookMlx`

MLX-native version of `BacktrackSteeringHook` from `sample.py`.

**Location:** `src/tgirl/sample_mlx.py`

Mirror the torch version but use `mx.array` operations. Must conform to `InferenceHookMlx` protocol (receives `valid_mask: mx.array` instead of `grammar_state`).

**Test Command:** `pytest tests/test_state_machine.py -v -k BacktrackSteeringHookMlx`

### Task 3: Best-so-far tracking on Checkpoint model

Extend the `Checkpoint` model in `state_machine.py` to track the best token sequence seen across all attempts from this checkpoint.

Add fields:
- `best_tokens: tuple[int, ...] = ()` — best sequence seen so far
- `best_mean_log_prob: float = float('-inf')` — mean log prob of best sequence
- `attempts: int = 0` — number of attempts from this checkpoint

Add method:
- `with_attempt(tokens: tuple[int, ...], mean_log_prob: float) -> Checkpoint` — returns new Checkpoint with updated best if better

**Test Command:** `pytest tests/test_state_machine.py -v -k best_so_far`

### Task 4: Full backtrack dispatch loop

Wire the backtrack loop into `run_constrained_generation` (torch) and `run_constrained_generation_mlx` (MLX).

**Modify signatures:**
```python
def run_constrained_generation(
    ...,
    confidence_monitor: ConstrainedConfidenceMonitor | None = None,
    grammar_guide_factory: Callable[[str], GrammarState] | None = None,
    grammar_text: str | None = None,
) -> ConstrainedGenerationResult:
```

**Loop logic:**
1. At each position, if `confidence_monitor.should_checkpoint(valid_count)`, save a Checkpoint
2. Record log probs via `confidence_monitor.record_log_prob()`
3. If `confidence_monitor.should_backtrack()` and `backtracks_remaining > 0`:
   a. Create `BacktrackEvent`
   b. Get the divergent token from the last checkpoint position
   c. Update checkpoint with `with_added_dead_end(token_id)`
   d. Return a `ConstrainedGenerationResult` with a new field `backtrack_requested: bool = False`
4. The caller (SamplingSession.run) handles the actual replay:
   - Create fresh grammar state from `grammar_text`
   - Replay `advance()` for tokens up to checkpoint position
   - Add `BacktrackSteeringHook` with the dead-end tokens
   - Re-run constrained generation from checkpoint

**Add to ConstrainedGenerationResult:**
- `backtrack_requested: bool = False`
- `backtrack_checkpoint: Checkpoint | None = None`
- `backtrack_events: list[Any] = []`

**Test Command:** `pytest tests/test_state_machine.py -v -k backtrack`

## Architectural Constraints

- `state_machine.py` has ZERO torch/mlx imports
- OT transport remains zero-coupled
- All existing 668 tests must continue to pass
- Both torch and MLX backends supported

## Test Command

```bash
pytest tests/ -v
```
