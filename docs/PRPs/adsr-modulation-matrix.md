# PRP: ADSR Modulation Matrix

## Source PRD: docs/PRDs/adsr-modulation-matrix.md
## Date: 2026-03-15

## 1. Context Summary

Replace the static inference hook pipeline with a configurable modulation matrix. An (11, 7) tensor routes source signals through a single `matmul` per token to produce all sampling parameter modulations. Source signals include grammar freedom, entropy, confidence, nesting depth, phase indicators, and cycle detection. Destinations include temperature, top_p, repetition bias, transport epsilon, opener bias, backtrack sensitivity, and presence penalty. The matrix subsumes `GrammarTemperatureHook` and `NestingDepthHook`.

Full design: `docs/design/adsr-envelope.md`

## 2. Codebase Analysis

### Interfaces to implement
- `InferenceHookMlx` protocol: `pre_forward(position, valid_mask, token_history, logits) -> ModelIntervention` (sample_mlx.py:73-88)
- Optional `advance(token_id)` and `reset()` (duck-typed, called in sampling loop at lines 529-531 and 413-415)

### Existing code to reuse
- `NestingDepthHookMlx._delta` dict and `_opener_ids` set construction (sample_mlx.py:119-177) — paren delta pre-computation from tokenizer vocab
- `_detect_cycle(tokens, max_period)` (sample.py:137-150) — suffix cycle detection
- `merge_interventions` (sample.py:95-103) — last-non-None-wins per field
- `apply_penalties_mlx` (sample_mlx.py:264-323) — reads logit_bias, repetition_penalty, presence_penalty, frequency_penalty
- `apply_shaping_mlx` (sample_mlx.py:326-366) — reads temperature, top_k, top_p

### Hook conversion pattern (sample.py:1014-1045)
```python
if isinstance(hook, GrammarTemperatureHook):
    self._mlx_hooks.append(GrammarTemperatureHookMlx(...))
elif isinstance(hook, RepetitionPenaltyHook):
    self._mlx_hooks.append(RepetitionPenaltyHookMlx(...))
elif isinstance(hook, NestingDepthHook):
    self._mlx_hooks.append(NestingDepthHookMlx(...))
```

New entry needed for `ModMatrixHook` → `ModMatrixHookMlx`.

### Intervention application order
1. Hooks produce `ModelIntervention` (position 3 in loop)
2. `apply_penalties_mlx` reads: repetition_penalty, presence_penalty, frequency_penalty, logit_bias (pre-OT)
3. `redistribute_logits_mlx` uses `TransportConfig` (currently static per-session)
4. `apply_shaping_mlx` reads: temperature, top_k, top_p (post-OT)

### Signals available per token
- `valid_mask` → `grammar_freedom` = `mx.sum(valid_mask).item() / vocab_size`
- `logits` → `token_entropy` via `mx.softmax` + `mx.log` + `mx.sum`
- `token_history[-1]` → `token_log_prob` (from previous token's telemetry, or re-compute)
- `grammar_mask_overlap` = `mx.sum(mx.softmax(logits) * valid_mask).item()`
- `position` → `position_normalized` = `position / max_tokens`
- `_depth` (from advance tracking) → `normalized_depth` = `depth / max_depth`
- Phase gates computed from depth + freedom transitions

## 3. Implementation Plan

**Test Command:** `pytest tests/test_modulation.py -v`

### Task 1: Source conditioning and phase detection

**Files:** `src/tgirl/modulation.py` (create), `tests/test_modulation.py` (create)

**Approach:**

```python
# modulation.py — pure Python + dataclasses, no framework deps

@dataclass(frozen=True)
class SourceConditionerConfig:
    """Per-source preprocessing config."""
    range_min: float = 0.0
    range_max: float = 1.0
    rectify: bool = False
    invert: bool = False
    slew_rate: float = 1.0  # 1.0 = no smoothing, 0.1 = heavy smoothing

@dataclass
class EnvelopeState:
    """Mutable per-generation state for phase detection."""
    phase: str = "attack"  # attack | decay | sustain | release
    phase_position: int = 0
    depth: int = 0
    prev_depth: int = 0
    prev_freedom: float = 1.0
    peak_freedom: float = 0.0
    prev_smoothed: list[float]  # per-source smoothed values

    def detect_phase(self, freedom: float, depth: int) -> str:
        """Detect ADSR phase from structural signals."""
        ...

    def advance_phase(self, freedom: float, depth: int) -> None:
        """Update phase state after a token."""
        ...

def condition_source(
    raw: float,
    cfg: SourceConditionerConfig,
    prev_smoothed: float,
) -> float:
    """Normalize, rectify, and slew-limit a single source value."""
    normalized = (raw - cfg.range_min) / max(cfg.range_max - cfg.range_min, 1e-10)
    normalized = max(0.0, min(1.0, normalized))
    if cfg.invert:
        normalized = 1.0 - normalized
    if cfg.rectify:
        normalized = max(0.0, normalized)
    return cfg.slew_rate * normalized + (1.0 - cfg.slew_rate) * prev_smoothed

DEFAULT_CONDITIONERS: tuple[SourceConditionerConfig, ...] = (
    SourceConditionerConfig(range_min=0.0, range_max=1.0),                    # 0: grammar_freedom
    SourceConditionerConfig(range_min=0.0, range_max=12.5),                   # 1: entropy (log(248k) ≈ 12.4)
    SourceConditionerConfig(range_min=-5.0, range_max=0.0, invert=True),      # 2: confidence (inverted: high=uncertain)
    SourceConditionerConfig(range_min=0.0, range_max=1.0),                    # 3: mask_overlap
    SourceConditionerConfig(range_min=0.0, range_max=10.0),                   # 4: depth
    SourceConditionerConfig(range_min=0.0, range_max=1.0),                    # 5: position
    SourceConditionerConfig(),  # 6: phase_attack (already 0 or 1)
    SourceConditionerConfig(),  # 7: phase_decay
    SourceConditionerConfig(),  # 8: phase_sustain
    SourceConditionerConfig(),  # 9: phase_release
    SourceConditionerConfig(),  # 10: cycle_detected
)
```

**Tests:**
- `condition_source` normalizes values to [0, 1] for various ranges
- `condition_source` with `invert=True` flips polarity
- `condition_source` with `rectify=True` clamps negative to 0
- `condition_source` with `slew_rate=0.5` smooths over 2 calls
- `detect_phase` returns "attack" when depth increases
- `detect_phase` returns "decay" when freedom collapses >70% from peak
- `detect_phase` returns "release" when depth decreases
- `detect_phase` returns "sustain" after stable decay period
- Phase hysteresis: flickering freedom doesn't cause rapid phase oscillation

**Validation:** `pytest tests/test_modulation.py -v`

### Task 2: Default modulation matrix and EnvelopeConfig

**Files:** `src/tgirl/modulation.py`, `tests/test_modulation.py`

**Approach:**

```python
@dataclass(frozen=True)
class EnvelopeConfig:
    """Complete modulation matrix configuration."""
    # Base values (destinations start here, modulations are added)
    base_temperature: float = 0.3
    base_top_p: float = 0.9
    base_repetition_bias: float = -20.0
    base_epsilon: float = 0.1
    base_opener_bias: float = 0.0
    base_backtrack_threshold: float = -0.5
    base_presence_penalty: float = 0.0

    # Output clamp ranges
    temperature_range: tuple[float, float] = (0.0, 2.0)
    top_p_range: tuple[float, float] = (0.1, 1.0)
    repetition_bias_range: tuple[float, float] = (-100.0, 0.0)
    epsilon_range: tuple[float, float] = (0.01, 1.0)

    # Source conditioners (11 entries)
    conditioners: tuple[SourceConditionerConfig, ...] = DEFAULT_CONDITIONERS

    # The matrix itself — shape (11, 7), stored as flat tuple for immutability
    matrix_flat: tuple[float, ...] = DEFAULT_MATRIX_FLAT

    @property
    def matrix_shape(self) -> tuple[int, int]:
        return (11, 7)

DEFAULT_MATRIX = [
    # temp   top_p  rep    eps    open   back   pres
    [ 0.3,   0.2,   0.0,   0.0,   0.0,   0.0,   0.0],  # grammar_freedom
    [ 0.1,   0.1,   0.0,   0.0,   0.0,  -0.2,   0.0],  # entropy
    [ 0.0,   0.0,   0.0,   0.0,   0.0,  -0.3,   0.0],  # confidence (inverted)
    [ 0.0,   0.0,   0.0,  -0.1,   0.0,   0.0,   0.0],  # mask_overlap
    [ 0.0,   0.0,   0.0,   0.0, -20.0,   0.0,   0.0],  # depth
    [ 0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0],  # position
    [ 0.3,   0.35,  10.0, -0.05,  0.0,   0.0,   0.0],  # phase_attack
    [ 0.0,   0.2,   5.0,   0.0,   0.0,  -0.1,   0.0],  # phase_decay
    [-0.15, -0.1, -10.0,   0.1,   0.0,   0.0,   0.0],  # phase_sustain
    [-0.05,  0.3,  10.0, -0.05, -50.0,   0.0,   0.0],  # phase_release
    [ 0.0,   0.0, -30.0,   0.0,   0.0,   0.0,   0.0],  # cycle_detected
]
```

**Tests:**
- Default matrix has correct shape (11, 7)
- Base values are sensible defaults
- Matrix weights match the ADSR design doc values
- Clamping ranges prevent invalid parameter values
- Config is frozen (immutable)

**Validation:** `pytest tests/test_modulation.py -v`

### Task 3: ModMatrixHookMlx — the unified hook

**Files:** `src/tgirl/modulation.py`, `tests/test_modulation.py`

**Approach:**

```python
class ModMatrixHookMlx:
    """Phase-aware modulation matrix for constrained generation.

    Routes 11 source signals through an (11, 7) matrix to produce
    7 parameter modulations per token. Replaces GrammarTemperatureHook
    and NestingDepthHook with a single configurable hook.
    """

    def __init__(
        self,
        config: EnvelopeConfig,
        tokenizer_decode: Callable[[list[int]], str],
        vocab_size: int,
        max_tokens: int = 128,
    ) -> None:
        self._config = config
        self._max_tokens = max_tokens
        self._state = EnvelopeState(prev_smoothed=[0.0] * 11)
        self._mod_matrix: mx.array  # shape (11, 7), constructed from config
        # Pre-compute paren deltas (reuse NestingDepthHook pattern)
        self._delta: dict[int, int] = {}
        self._opener_ids: set[int] = set()
        # ... vocab scan as in NestingDepthHookMlx.__init__

    def reset(self) -> None:
        """Reset state for new constrained generation pass."""
        self._state = EnvelopeState(prev_smoothed=[0.0] * 11)

    def advance(self, token_id: int) -> None:
        """Update depth and phase after token sampled."""
        self._state.depth += self._delta.get(token_id, 0)
        self._state.depth = max(self._state.depth, 0)

    def pre_forward(
        self,
        position: int,
        valid_mask: mx.array,
        token_history: list[int],
        logits: mx.array,
    ) -> ModelIntervention:
        """Compute modulated parameters via matrix multiply."""
        vocab_size = logits.shape[0]

        # 1. Compute raw source signals
        freedom = float(mx.sum(valid_mask).item()) / vocab_size
        probs = mx.softmax(logits, axis=-1)
        entropy = -float(mx.sum(probs * mx.log(mx.clip(probs, 1e-30, None))).item())
        overlap = float(mx.sum(probs * valid_mask.astype(mx.float32)).item())
        confidence = float(mx.max(mx.log(mx.clip(probs, 1e-30, None))).item())

        # 2. Detect phase
        self._state.advance_phase(freedom, self._state.depth)
        phase = self._state.phase

        # 3. Cycle detection
        cycle = 1.0 if _detect_cycle(token_history) is not None else 0.0

        # 4. Build source vector (conditioned)
        raw_sources = [
            freedom, entropy, confidence, overlap,
            float(self._state.depth), position / self._max_tokens,
            1.0 if phase == "attack" else 0.0,
            1.0 if phase == "decay" else 0.0,
            1.0 if phase == "sustain" else 0.0,
            1.0 if phase == "release" else 0.0,
            cycle,
        ]
        conditioned = []
        for i, (raw, cfg) in enumerate(zip(raw_sources, self._config.conditioners)):
            val = condition_source(raw, cfg, self._state.prev_smoothed[i])
            self._state.prev_smoothed[i] = val
            conditioned.append(val)

        # 5. Matrix multiply (native MLX)
        source_vec = mx.array(conditioned)
        modulations = source_vec @ self._mod_matrix  # shape (7,)

        # 6. Apply base values + clamp
        cfg = self._config
        temperature = max(cfg.temperature_range[0], min(cfg.temperature_range[1],
            cfg.base_temperature + float(modulations[0].item())))
        top_p = max(cfg.top_p_range[0], min(cfg.top_p_range[1],
            cfg.base_top_p + float(modulations[1].item())))
        rep_bias = float(modulations[2].item()) + cfg.base_repetition_bias
        epsilon = max(cfg.epsilon_range[0], min(cfg.epsilon_range[1],
            cfg.base_epsilon + float(modulations[3].item())))
        opener_bias_val = cfg.base_opener_bias + float(modulations[4].item())
        presence = cfg.base_presence_penalty + float(modulations[6].item())

        # 7. Build logit_bias for opener penalty (if any)
        logit_bias = None
        if opener_bias_val < -1.0:
            logit_bias = {tid: opener_bias_val for tid in self._opener_ids}

        # 8. Build intervention
        # Repetition bias applied via logit_bias on repeated tokens
        if rep_bias < cfg.base_repetition_bias:
            # Merge with opener bias
            recent = token_history[-8:]
            counts: dict[int, int] = {}
            for tid in recent:
                counts[tid] = counts.get(tid, 0) + 1
            for tid, count in counts.items():
                if count > 2:
                    bias = rep_bias * (count - 2)
                    if logit_bias is None:
                        logit_bias = {}
                    existing = logit_bias.get(tid, 0.0)
                    logit_bias[tid] = min(existing + bias, -100.0)

        return ModelIntervention(
            temperature=temperature,
            top_p=top_p,
            logit_bias=logit_bias,
            transport_epsilon=epsilon,
            presence_penalty=presence if presence != 0.0 else None,
        )
```

**Tests:**
- Hook implements InferenceHookMlx protocol
- pre_forward returns ModelIntervention with temperature, top_p, logit_bias
- Phase detection affects output (attack → higher temperature than sustain)
- Depth tracking via advance() matches NestingDepthHook behavior
- Opener penalty activates when depth + position approaches budget
- Cycle detection gate activates repetition bias
- Slew rate smoothing prevents parameter discontinuities
- Matrix multiply uses native mx.matmul (verify via source inspection)
- Hook reset() clears all state
- Per-token overhead <0.1ms (timing test)

**Validation:** `pytest tests/test_modulation.py -v`

### Task 4: ModMatrixHook (torch variant) + auto-conversion

**Files:** `src/tgirl/modulation.py`, `src/tgirl/sample.py`, `tests/test_modulation.py`

**Approach:**

Torch variant mirrors MLX but uses `torch.matmul`, `torch.softmax`, etc. Same `EnvelopeConfig`, same `EnvelopeState` (pure Python), different tensor ops in `pre_forward`.

Auto-conversion in `SamplingSession` (sample.py):
```python
elif isinstance(hook, ModMatrixHook):
    self._mlx_hooks.append(
        ModMatrixHookMlx(
            config=hook._config,
            tokenizer_decode=self._decode,
            vocab_size=self._embeddings_mlx.shape[0],
            max_tokens=hook._max_tokens,
        )
    )
```

**Tests:**
- Torch variant produces same outputs as MLX for identical inputs (numerical tolerance)
- Auto-conversion in SamplingSession works
- No cross-framework conversions in either variant

**Validation:** `pytest tests/test_modulation.py -v`

### Task 5: transport_epsilon on ModelIntervention + sampling loop integration

**Files:** `src/tgirl/types.py`, `src/tgirl/sample_mlx.py`, `src/tgirl/sample.py`, `tests/test_modulation.py`

**Approach:**

Add `transport_epsilon: float | None = None` to `ModelIntervention`.

In both sampling loops, after merging interventions and before OT:
```python
# Read per-token epsilon if set by a hook
if merged.transport_epsilon is not None:
    token_transport_config = TransportConfig(
        epsilon=merged.transport_epsilon,
        max_iterations=transport_config.max_iterations,
        convergence_threshold=transport_config.convergence_threshold,
        valid_ratio_threshold=transport_config.valid_ratio_threshold,
        invalid_mass_threshold=transport_config.invalid_mass_threshold,
        max_problem_size=transport_config.max_problem_size,
    )
else:
    token_transport_config = transport_config

ot_result = redistribute_logits_mlx(adjusted, valid_mask, embeddings, config=token_transport_config)
```

**Tests:**
- `ModelIntervention(transport_epsilon=0.05)` is valid
- Sampling loop uses per-token epsilon when set
- Sampling loop uses session-level epsilon when not set
- Default behavior unchanged (transport_epsilon=None)

**Validation:** `pytest tests/test_modulation.py tests/test_sample_mlx.py -v`

### Task 6: Wire into benchmarks and showcase

**Files:** `benchmarks/run_bfcl.py`, `examples/showcase_unified_api.py`

**Approach:**

Replace:
```python
session_hooks = [
    GrammarTemperatureHook(base_temperature=0.5),
    RepetitionPenaltyHook(window=8, max_repeats=2),
    NestingDepthHook(max_tokens=128, tokenizer_decode=..., vocab_size=...),
]
```

With:
```python
from tgirl.modulation import ModMatrixHook, EnvelopeConfig

session_hooks = [
    ModMatrixHook(
        config=EnvelopeConfig(base_temperature=0.5),
        tokenizer_decode=hf_tokenizer.decode,
        vocab_size=embeddings.shape[0],
        max_tokens=128,
    ),
    RepetitionPenaltyHook(window=8, max_repeats=2),  # kept for cycle detection
]
```

**Tests:**
- BFCL benchmark runs without errors with ModMatrixHook
- Showcase runs without errors
- Accuracy within 5% of baseline (may improve or slightly regress — record both)

**Validation:** `pytest tests/ -v && python -u examples/showcase_unified_api.py`

### Task 7: Telemetry integration

**Files:** `src/tgirl/modulation.py`, `src/tgirl/sample_mlx.py`

**Approach:**

Add per-token envelope telemetry to the constrained generation result:
```python
@dataclass
class EnvelopeTelemetry:
    phase: str
    phase_position: int
    depth: int
    source_vector: list[float]    # 11 conditioned values
    modulation_vector: list[float]  # 7 output values
    final_temperature: float
    final_epsilon: float
```

Record in the sampling loop alongside existing telemetry. Store as part of `ConstrainedGenerationResult` or in a parallel list.

**Tests:**
- Telemetry records phase, depth, and vectors
- Telemetry is JSON-serializable
- Telemetry list length matches token count

**Validation:** `pytest tests/test_modulation.py -v`

## 4. Validation Gates

```bash
# Lint
ruff check src/tgirl/modulation.py

# Unit tests
pytest tests/test_modulation.py -v

# Full suite (no regressions)
pytest tests/ -v --ignore=tests/test_cache.py \
    --ignore=tests/test_transport.py --ignore=tests/test_transport_mlx.py

# Benchmark comparison (before/after)
python -u benchmarks/run_bfcl.py --model mlx-community/Qwen3.5-0.8B-MLX-4bit \
    --category simple_python --limit 50 --model-name tgirl-modmatrix
```

## 5. Rollback Plan

`modulation.py` is a new file. `transport_epsilon` on `ModelIntervention` is additive (default None). Rollback: delete `modulation.py`, remove the field, revert sampling loop epsilon reads, restore old hook lists in benchmarks/showcase.

## 6. Uncertainty Log

- **Source signal computation inside pre_forward:** Computing `mx.softmax(logits)` and entropy inside the hook duplicates work that the CSG also does. Consider whether the hook should receive the `TransitionSignal` directly instead of recomputing from raw logits. This would require changing the `InferenceHookMlx` protocol — defer to a follow-up if the performance cost is negligible (<0.1ms).
- **Phase detection hysteresis:** The design doc mentions hysteresis but doesn't specify the exact mechanism. The implementation should require ≥2 consecutive tokens meeting the transition condition before switching phases.
- **Shaped phase ramps vs binary gates:** The design doc describes both options. V1 uses binary gates for simplicity; shaped ramps are a v2 enhancement.
- **RepetitionPenaltyHook coexistence:** The mod matrix handles repetition bias via the envelope, but `_detect_cycle` integration means the cycle gate in the matrix partially overlaps with the separate RepetitionPenaltyHook. For v1, keep both — the matrix handles phase-aware bias scaling, the separate hook handles the window-based counting and cycle detection. Merge in v2.
