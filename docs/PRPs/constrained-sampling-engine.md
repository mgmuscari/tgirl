# PRP: Constrained Sampling Engine

## Source PRD: docs/PRDs/constrained-sampling-engine.md
## Date: 2026-03-12

## 1. Context Summary

Implement `tgirl.sample` — the keystone module that integrates grammar constraints (from `tgirl.grammar`), optimal transport redistribution (from `tgirl.transport`), and Hy pipeline execution (from `tgirl.compile`) into a dual-mode sampling loop. The module manages inference sessions that alternate between freeform natural language generation and grammar-constrained Hy s-expression generation, with per-token hook interventions and telemetry recording.

## 2. Codebase Analysis

### Relevant Existing Patterns

- **Frozen Pydantic models** throughout `types.py` (lines 24-254): All data types use `model_config = ConfigDict(frozen=True)`. New types (`SessionConfig`, `ModelIntervention`) must follow this pattern.
- **NamedTuple for results** in `transport.py:35-46`: `TransportResult` uses `NamedTuple` for lightweight positional + named access. Sample results could follow this or use frozen Pydantic (prefer Pydantic for consistency with other output types).
- **`structlog` logging** in all modules: `logger = structlog.get_logger()` at module level. Debug-level logging for internal operations, info-level for completed actions.
- **Config objects with defaults**: `TransportConfig`, `GrammarConfig`, `CompileConfig` all use frozen Pydantic with sensible defaults. `SessionConfig` follows this pattern.
- **`__init__.py` re-exports**: All public types are re-exported from `src/tgirl/__init__.py` with `__all__`.
- **Test organization**: Tests mirror module structure. Class-based grouping by task (e.g., `TestTransportConfig` for Task 1). Imports inside test methods for isolation.

### Integration Points

- `tgirl.grammar.generate(snapshot, config) → GrammarOutput` (`grammar.py:382-428`): Produces Lark EBNF text and metadata from a `RegistrySnapshot`.
- `tgirl.transport.redistribute_logits(logits, valid_mask, embeddings, config=...) → TransportResult` (`transport.py:223-345`): OT redistribution. Requires `torch.Tensor` logits, boolean valid mask, and embedding matrix.
- `tgirl.compile.run_pipeline(source, registry, config) → PipelineResult | InsufficientResources | PipelineError` (`compile.py:702-831`): Compiles and executes Hy source in sandbox.
- `tgirl.registry.ToolRegistry.snapshot(scopes=..., cost_budget=...) → RegistrySnapshot` (`registry.py:105-152`): Produces immutable registry state.
- `tgirl.types.TelemetryRecord` (`types.py:222-244`): Pre-defined telemetry schema.

### Conventions

- Branch naming: `feature/<slug>` (target branch: `feature/constrained-sampling-engine`, to be created from `main` at execution time)
- Commits: conventional commits, one per task (test + implementation together)
- TDD: RED → GREEN → REFACTOR for each task
- `ruff check` and `mypy` must pass

## 3. Implementation Plan

**Test Command:** `pytest tests/ -v`

### Task 1: Add SessionConfig and ModelIntervention to types.py

**Files:** `src/tgirl/types.py`, `tests/test_sample.py` (create)
**Approach:**

Add two frozen Pydantic models to `types.py`:

```python
class ModelIntervention(BaseModel):
    model_config = ConfigDict(frozen=True)
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[int, float] | None = None
    activation_steering: Any | None = None  # Reserved for ESTRADIOL

class SessionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    # Freeform mode
    freeform_temperature: float = 0.7
    freeform_top_p: float = 0.9
    freeform_top_k: int | None = None
    freeform_repetition_penalty: float = 1.0
    freeform_max_tokens: int = 4096
    # Constrained mode
    constrained_base_temperature: float = 0.3
    constrained_ot_epsilon: float = 0.1
    constrained_max_tokens: int = 512
    # Session-level
    max_tool_cycles: int = 10
    session_cost_budget: float | None = None
    session_timeout: float = 300.0
    # Delimiters
    tool_open_delimiter: str = "<tool>"
    tool_close_delimiter: str = "</tool>"
    result_open_delimiter: str = "<tool_result>"
    result_close_delimiter: str = "</tool_result>"
```

**Tests:**
- `SessionConfig` is frozen (mutation raises `ValidationError`)
- `SessionConfig` defaults match TGIRL.md section 3.4
- `ModelIntervention` is frozen
- `ModelIntervention` defaults to all None
- `SessionConfig` custom values accepted
- Delimiter fields are configurable

**Validation:** `pytest tests/test_sample.py -v -k "TestSessionConfig or TestModelIntervention"`

### Task 2: Define InferenceHook Protocol and hook merge function

**Files:** `src/tgirl/sample.py` (create), `tests/test_sample.py`
**Approach:**

Create `src/tgirl/sample.py` with:

```python
from typing import Protocol, Any
import torch
from tgirl.types import ModelIntervention

class GrammarState(Protocol):
    """Protocol for grammar state trackers."""
    def get_valid_mask(self, tokenizer_vocab_size: int) -> torch.Tensor: ...
    def is_accepting(self) -> bool: ...
    def advance(self, token_id: int) -> None: ...

class InferenceHook(Protocol):
    def pre_forward(
        self,
        position: int,
        grammar_state: GrammarState,
        token_history: list[int],
        logits: torch.Tensor,
    ) -> ModelIntervention: ...

def merge_interventions(interventions: list[ModelIntervention]) -> ModelIntervention:
    """Merge multiple hook interventions. Last non-None value wins per field."""
    merged = {}
    for intervention in interventions:
        for field_name in intervention.model_fields:
            val = getattr(intervention, field_name)
            if val is not None:
                merged[field_name] = val
    return ModelIntervention(**merged)
```

**Tests:**
- `merge_interventions` with empty list returns all-None `ModelIntervention`
- `merge_interventions` with single intervention preserves all values
- `merge_interventions` with two interventions: later overrides earlier (last-writer-wins)
- `merge_interventions` preserves non-overlapping fields from both
- A class implementing `InferenceHook` protocol is accepted by `isinstance` check (runtime_checkable)

**Validation:** `pytest tests/test_sample.py -v -k "TestMergeInterventions or TestInferenceHook"`

### Task 3: Implement GrammarTemperatureHook

**Files:** `src/tgirl/sample.py`, `tests/test_sample.py`
**Approach:**

```python
class GrammarTemperatureHook:
    """Default hook: grammar-implied temperature scheduling."""
    def __init__(self, base_temperature: float = 0.3, scaling_exponent: float = 0.5):
        self.base_temperature = base_temperature
        self.scaling_exponent = scaling_exponent

    def pre_forward(self, position, grammar_state, token_history, logits):
        vocab_size = logits.shape[-1]
        valid_mask = grammar_state.get_valid_mask(vocab_size)
        valid_count = valid_mask.sum().item()
        if valid_count <= 1:
            return ModelIntervention(temperature=0.0)
        freedom = valid_count / vocab_size
        temp = self.base_temperature * (freedom ** self.scaling_exponent)
        return ModelIntervention(temperature=temp)
```

**Tests:**
- `valid_count == 1` → temperature 0.0
- `valid_count == 0` → temperature 0.0
- `valid_count == vocab_size` → temperature == `base_temp * 1.0^0.5 == base_temp`
- `valid_count == vocab_size // 4` → temperature == `base_temp * sqrt(0.25)`
- Custom `scaling_exponent=1.0` → linear scaling
- Custom `base_temperature` is respected

**Validation:** `pytest tests/test_sample.py -v -k "TestGrammarTemperatureHook"`

### Task 4: Implement logit processing pipeline (split pre-OT and post-OT)

**Files:** `src/tgirl/sample.py`, `tests/test_sample.py`
**Approach:**

Per TGIRL.md section 8.5 (steps d-f), logit processing is split into two phases around OT redistribution. Penalties operate on raw logits (preserving the model's intent signal for OT), while shaping operates on the redistributed distribution.

```python
def apply_penalties(
    logits: torch.Tensor,
    intervention: ModelIntervention,
    token_history: list[int],
) -> torch.Tensor:
    """Pre-OT: apply repetition, presence, frequency penalties and logit bias to raw logits.

    These modify the model's original logit distribution before OT redistribution,
    so that OT operates on a penalty-adjusted signal (TGIRL.md 8.5 steps e-f ordering
    note: penalties logically precede OT even though the spec lists them after shaping —
    see rationale below).
    """
    result = logits.clone()

    # Repetition penalty
    if intervention.repetition_penalty is not None and intervention.repetition_penalty != 1.0:
        for token_id in set(token_history):
            if result[token_id] > 0:
                result[token_id] /= intervention.repetition_penalty
            else:
                result[token_id] *= intervention.repetition_penalty

    # Presence penalty
    if intervention.presence_penalty is not None and intervention.presence_penalty != 0.0:
        for token_id in set(token_history):
            result[token_id] -= intervention.presence_penalty

    # Frequency penalty
    if intervention.frequency_penalty is not None and intervention.frequency_penalty != 0.0:
        from collections import Counter
        counts = Counter(token_history)
        for token_id, count in counts.items():
            result[token_id] -= intervention.frequency_penalty * count

    # Logit bias
    if intervention.logit_bias is not None:
        for token_id, bias in intervention.logit_bias.items():
            result[token_id] += bias

    return result


def apply_shaping(
    logits: torch.Tensor,
    intervention: ModelIntervention,
) -> torch.Tensor:
    """Post-OT: apply temperature, top-k, top-p to redistributed logits.

    These shape the final sampling distribution after OT has redistributed
    probability mass onto valid tokens (TGIRL.md 8.5 step e).
    """
    result = logits.clone()

    # Temperature
    if intervention.temperature is not None and intervention.temperature > 0:
        result = result / intervention.temperature
    elif intervention.temperature is not None and intervention.temperature == 0:
        # Greedy: set all but max to -inf. Tie-breaking policy: torch.argmax
        # returns the first (lowest-index) maximum. This is deterministic and
        # consistent, which is the desired property for greedy decoding.
        max_idx = result.argmax()
        mask = torch.ones_like(result, dtype=torch.bool)
        mask[max_idx] = False
        result[mask] = float('-inf')

    # Top-k
    if intervention.top_k is not None and intervention.top_k > 0:
        top_k_vals, _ = torch.topk(result, min(intervention.top_k, result.shape[-1]))
        threshold = top_k_vals[-1]
        result[result < threshold] = float('-inf')

    # Top-p (nucleus)
    if intervention.top_p is not None and intervention.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(result, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        cutoff_mask = cumulative - probs > intervention.top_p
        sorted_logits[cutoff_mask] = float('-inf')
        # Unsort: scatter modified sorted values back to original positions
        result = torch.empty_like(sorted_logits)
        result.scatter_(0, sorted_indices, sorted_logits)

    return result
```

**Ordering rationale:** The spec (TGIRL.md 8.5) lists steps as d=OT, e=temperature/top-p/top-k, f=penalties. However, applying penalties *after* OT means OT redistributes mass onto tokens the model intended to penalize, which the penalties then pull back — wasting OT computation. Applying penalties *before* OT lets OT redistribute from an already-adjusted distribution. The PRP places penalties pre-OT and shaping post-OT. This deviation from the spec's literal ordering is logged in the Uncertainty Log and should be validated during implementation.

**Tests:**
- `apply_penalties`: repetition penalty > 1.0 penalizes repeated tokens
- `apply_penalties`: presence penalty subtracts from seen tokens
- `apply_penalties`: frequency penalty scales with count
- `apply_penalties`: logit bias shifts specific tokens
- `apply_penalties`: all-None intervention returns logits unchanged
- `apply_shaping`: temperature 1.0 leaves logits unchanged (within float tolerance)
- `apply_shaping`: temperature 0.5 doubles logits
- `apply_shaping`: temperature 0.0 is greedy (only max survives)
- `apply_shaping`: temperature 0.0 with tied maxima → lowest-index token wins (documents tie-breaking policy)
- `apply_shaping`: top-k=3 keeps only top 3
- `apply_shaping`: top-p=0.5 with known logits [5.0, 3.0, 1.0, 0.5, 0.1] keeps tokens whose cumulative softmax probability <= 0.5, sets rest to -inf, and result indices match original (unsorted) positions
- `apply_shaping`: all-None intervention returns logits unchanged

**Validation:** `pytest tests/test_sample.py -v -k "TestApplyPenalties or TestApplyShaping"`

### Task 5: Implement MockGrammarState and constrained token generation core

**Files:** `src/tgirl/sample.py`, `tests/test_sample.py`
**Approach:**

Build the constrained generation core that, given a grammar state, logits source (callable), embeddings, hooks, and transport config, generates tokens until grammar accepts.

**Per-token processing order** (follows TGIRL.md 8.5 with penalties moved pre-OT — see Task 4 rationale):

```
1. forward_fn(context_tokens) → raw logits
2. grammar_state.get_valid_mask(vocab_size) → valid_mask
3. call all InferenceHooks → merge_interventions() → merged ModelIntervention
4. apply_penalties(logits, intervention, token_history) → penalty-adjusted logits
5. redistribute_logits(adjusted_logits, valid_mask, embeddings, config) → OT result
6. apply_shaping(ot_logits, intervention) → shaped logits
7. sample token from shaped logits
8. grammar_state.advance(token_id)
9. record telemetry
```

```python
class ConstrainedGenerationResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    tokens: list[int]
    hy_source: str
    grammar_valid_counts: list[int]
    temperatures_applied: list[float]
    wasserstein_distances: list[float]
    top_p_applied: list[float]
    token_log_probs: list[float]
    ot_computation_total_ms: float
    ot_bypassed_count: int
    grammar_generation_ms: float

def run_constrained_generation(
    grammar_state: GrammarState,
    forward_fn: Callable[[list[int]], torch.Tensor],
    tokenizer_decode: Callable[[list[int]], str],
    embeddings: torch.Tensor,
    hooks: list[InferenceHook],
    transport_config: TransportConfig,
    max_tokens: int = 512,
    context_tokens: list[int] | None = None,
) -> ConstrainedGenerationResult:
    ...
```

For testing, create a `MockGrammarState` that feeds a predetermined sequence of valid masks and accepting states, and a mock forward function that returns controlled logits.

**Tests:**
- With a mock that accepts after 3 tokens with forced valid masks → produces 3 tokens
- Hooks are called at each position in order
- Processing order is enforced: penalties → OT → shaping (verify via mock side effects)
- OT redistribution is called (or bypassed) at each position
- Telemetry fields (valid counts, temperatures, wasserstein) are populated
- `max_tokens` limit is respected (stops if grammar never accepts)
- Token log probs are recorded

**Validation:** `pytest tests/test_sample.py -v -k "TestConstrainedGeneration"`

### Task 6: Implement delimiter detection for mode switching

**Files:** `src/tgirl/sample.py`, `tests/test_sample.py`
**Approach:**

```python
class DelimiterDetector:
    """Detects tool call delimiters in generated token stream.

    Maintains a sliding window of decoded text rather than accumulating
    all token IDs. The window is bounded to 2x the delimiter's character
    length, which is sufficient to detect any delimiter that spans a
    token boundary.
    """

    def __init__(self, delimiter: str, tokenizer_decode: Callable[[list[int]], str]):
        self.delimiter = delimiter
        self.decode = tokenizer_decode
        self._decoded_window: str = ""
        self._max_window = len(delimiter) * 2

    def feed(self, token_id: int) -> bool:
        """Feed a token. Returns True if delimiter is detected."""
        new_text = self.decode([token_id])
        self._decoded_window += new_text
        # Prune window to bounded size, keeping enough trailing chars
        # to detect a delimiter that spans the pruning boundary
        if len(self._decoded_window) > self._max_window:
            self._decoded_window = self._decoded_window[-self._max_window:]
        return self.delimiter in self._decoded_window

    def reset(self) -> None:
        self._decoded_window = ""
```

**Design note:** Decoding single tokens individually avoids accumulating a token ID buffer entirely. The decoded text window is bounded to `2 * len(delimiter)` characters — enough to catch any cross-token-boundary delimiter occurrence. This trades per-token decode calls (cheap for single tokens) for guaranteed O(1) memory regardless of session length.

**Tests:**
- Single-token delimiter → detected immediately
- Multi-token delimiter → detected after final token
- No delimiter in stream → never triggers
- Reset clears state
- Partial match followed by non-match → no false positive
- Buffer stays bounded after 10,000 tokens (window length <= `2 * len(delimiter)`)

**Validation:** `pytest tests/test_sample.py -v -k "TestDelimiterDetector"`

### Task 7: Implement SamplingSession (orchestrator)

**Files:** `src/tgirl/sample.py`, `tests/test_sample.py`
**Approach:**

The main session class that orchestrates the dual-mode loop:

```python
class SamplingSession:
    def __init__(
        self,
        registry: ToolRegistry,
        forward_fn: Callable[[list[int]], torch.Tensor],
        tokenizer_decode: Callable[[list[int]], str],
        tokenizer_encode: Callable[[str], list[int]],
        embeddings: torch.Tensor,
        grammar_guide_factory: Callable[[str], GrammarState],
        config: SessionConfig | None = None,
        hooks: list[InferenceHook] | None = None,
        transport_config: TransportConfig | None = None,
    ):
        ...

    def run(self, prompt_tokens: list[int]) -> SamplingResult:
        """Run the full dual-mode sampling loop."""
        ...
```

The session:
1. Generates tokens in freeform mode using freeform params
2. Detects tool delimiter → switches to constrained mode
3. Generates grammar, creates grammar state, runs constrained generation
4. Passes completed Hy source to `run_pipeline()`
5. Updates quota state and injects results into context, returns to freeform
6. Respects cycle limits, cost budget, timeout

**Cross-cycle quota tracking (TGIRL.md 3.3):**

The session maintains a mutable `_consumed_quotas: dict[str, int]` counter. After each `run_pipeline()` call, the session counts tool invocations in the Hy source by walking the Hy AST for function call names that match registered tool names. When generating the snapshot for the next cycle, the session constructs a `RegistrySnapshot` with reduced quotas:

```python
def _snapshot_with_remaining_quotas(self) -> RegistrySnapshot:
    """Produce a snapshot with quotas reduced by consumed counts."""
    base = self._registry.snapshot(scopes=self._scopes, cost_budget=self._cost_remaining)
    remaining_quotas = {
        name: max(0, limit - self._consumed_quotas.get(name, 0))
        for name, limit in base.quotas.items()
    }
    return RegistrySnapshot(
        tools=base.tools,
        quotas=remaining_quotas,
        cost_remaining=self._cost_remaining,
        scopes=base.scopes,
        timestamp=base.timestamp,
    )

def _count_tool_invocations(self, hy_source: str) -> dict[str, int]:
    """Count tool invocations in Hy source via simple AST walk.

    This is a lightweight parse — not a full compile — that counts
    function call names matching registered tools.
    """
    ...
```

This means the grammar for cycle N+1 has tighter quotas than cycle N, enforcing the per-session quota invariant at the grammar level (safety by construction). Tools that have exhausted their quota become inexpressible in subsequent cycles.

```python
class SamplingResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    output_text: str
    tool_calls: list[ToolCallRecord]
    telemetry: list[TelemetryRecord]
    total_tokens: int
    total_cycles: int
    wall_time_ms: float
    quotas_consumed: dict[str, int]  # Final consumed quota counts

class ToolCallRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    pipeline: str  # Hy source
    result: Any
    error: PipelineError | None = None
    cycle_number: int
    tool_invocations: dict[str, int]  # Per-tool invocation counts for this cycle
```

**Tests:**
- Session with no tool call delimiter in freeform → returns freeform output only, zero cycles
- Session with one tool call cycle → freeform + constrained + result injection + more freeform
- `max_tool_cycles` enforced → stops after limit
- `session_timeout` enforced → stops if exceeded
- Session produces `TelemetryRecord` for each constrained cycle
- Two-cycle session: tool with quota=2 invoked once per cycle → second cycle snapshot has quota=1, third cycle (if any) would have quota=0 and tool becomes inexpressible
- `_count_tool_invocations` correctly counts nested/composed tool calls in `(-> (foo x) (bar))` as foo=1, bar=1
- `SamplingResult.quotas_consumed` reflects total consumed quotas across all cycles

**Validation:** `pytest tests/test_sample.py -v -k "TestSamplingSession"`

### Task 8: Wire up __init__.py exports and verify full integration

**Files:** `src/tgirl/__init__.py`, `tests/test_integration_sample.py` (create)
**Approach:**

Add exports to `__init__.py`:
- `SessionConfig`, `ModelIntervention` (from types)
- `InferenceHook`, `GrammarState` (protocols from sample)
- `GrammarTemperatureHook`, `SamplingSession`, `SamplingResult` (from sample)
- `merge_interventions`, `apply_penalties`, `apply_shaping`, `run_constrained_generation` (from sample)

Integration test: construct a minimal end-to-end test with a mock model that exercises the full pipeline: registry → grammar → constrained generation → compile → result. This test verifies the module boundaries work together.

**Tests:**
- All new public names importable from `tgirl`
- Integration test: register tools, run session with mock model, verify pipeline executes correctly
- `ruff check src/tgirl/sample.py src/tgirl/types.py` passes

**Validation:** `pytest tests/test_integration_sample.py -v && ruff check src/ --fix`

## 4. Validation Gates

```bash
# Syntax/Style
ruff check src/ --fix && ruff format src/

# Unit Tests
pytest tests/test_sample.py -v

# Integration Tests
pytest tests/test_integration_sample.py -v

# Full Suite (no regressions)
pytest tests/ -v

# Type Checking
mypy src/tgirl/sample.py src/tgirl/types.py
```

## 5. Rollback Plan

All changes are on `feature/constrained-sampling-engine`. If issues are discovered:
1. `types.py` changes are additive (new models only) — can be reverted without affecting existing modules
2. `sample.py` is a new file — deletion fully rolls back
3. `__init__.py` export additions are additive — revert the diff
4. No changes to existing module behavior (registry, grammar, compile, transport)

## 6. Uncertainty Log

1. **Outlines API surface:** The PRP assumes Outlines provides a way to create a grammar state from Lark EBNF and get valid token masks. The actual Outlines API may differ — Task 5 uses a `GrammarState` Protocol to abstract this, and the `grammar_guide_factory` parameter in Task 7 allows the caller to provide the Outlines integration. This isolates us from Outlines API changes.

2. **Token-level delimiter detection:** The `DelimiterDetector` in Task 6 decodes recent tokens to check for delimiters. This requires a tokenizer decode function. For multi-byte or BPE tokens, a delimiter might be split across token boundaries in unexpected ways. The buffer-based approach handles this but may have edge cases with very long delimiters.

3. **Top-p implementation:** The original scatter-based top-p had a bug (self-referential scatter). Fixed to use `torch.empty_like` + `scatter_()` for correct unsorting. Test now specifies exact numerical expectations with known input logits to verify both filtering and unsort correctness.

4. **Embedding matrix source:** The `SamplingSession` receives embeddings as a parameter. In practice, this comes from `model.get_input_embeddings().weight` (HuggingFace) or equivalent. This is documented but not enforced.

5. **`torch` as test dependency:** Tasks 2-8 require `torch`. The existing transport tests already use `torch`, so this dependency is available in the test environment. Verified via `pyproject.toml` optional deps.

6. **Penalty ordering deviation from spec:** TGIRL.md 8.5 lists OT (step d) before penalties (step f). The PRP places penalties pre-OT so that OT redistributes from a penalty-adjusted distribution rather than having penalties fight OT's redistribution. This is a deliberate deviation that should be validated empirically during implementation — if penalties post-OT produces better results, the ordering should be revised.
