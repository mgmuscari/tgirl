# ADSR Envelope Generator — Design Document

**Status:** Design
**Date:** 2026-03-15
**Architecture block:** SCU (Stochastic Control Unit) + CSG (Confidence Signal Generator)
**Metaphor:** The Linguistic Synthesis Engine generates structured language through oscillators (CLU), waveshapers (GSU), filters (STB), and envelope generators (this document).

---

## 1. The Problem with Static Parameters

Every inference hook today uses static parameters:

```python
GrammarTemperatureHook(base_temperature=0.5, scaling_exponent=0.5)
RepetitionPenaltyHook(window=8, max_repeats=2, bias=-20.0)
NestingDepthHook(max_tokens=128, margin=2, bias=-100.0)
```

These don't change over the lifecycle of an s-expression. But the lifecycle has distinct phases — opening a tool call is a different regime than filling in argument values, which is a different regime than closing nested parens. Static parameters are a rectangle wave applied where an ADSR envelope belongs.

---

## 2. ADSR Phases in Constrained Generation

The "note" is one constrained generation pass — from the moment the grammar activates to the moment it accepts.

### Phase Detection

The system already has the signals to detect phase. No new instrumentation needed.

| Phase | Structural Signal | CSG Signal |
|-------|-------------------|------------|
| **Attack** | `depth` increasing, `grammar_freedom` high | High entropy (many valid continuations) |
| **Decay** | `depth` stable, `grammar_freedom` dropping sharply | Entropy falling (tool name selected, args beginning) |
| **Sustain** | `depth` stable, `grammar_freedom` low-moderate | Entropy stable, `token_log_prob` high (model is confident about values) |
| **Release** | `depth` decreasing, approaching 0 | `grammar_freedom` may spike briefly at closing positions |

**Phase transition detection:**

```python
@dataclass
class EnvelopeState:
    phase: Literal["attack", "decay", "sustain", "release"]
    phase_position: int     # tokens since phase started
    depth: int              # current nesting depth
    prev_depth: int         # depth at previous token
    prev_freedom: float     # grammar_freedom at previous token
    peak_freedom: float     # max freedom seen in attack phase

def detect_phase(state: EnvelopeState, freedom: float, depth: int) -> str:
    if depth > state.prev_depth:
        # Opening new nesting level — attack
        return "attack"
    if state.phase == "attack" and freedom < state.peak_freedom * 0.3:
        # Freedom collapsed >70% from peak — tool selected, entering decay
        return "decay"
    if depth < state.prev_depth:
        # Closing nesting level — release
        return "release"
    if state.phase == "decay" and state.phase_position > 2:
        # Been in decay >2 tokens — stabilizing into sustain
        return "sustain"
    return state.phase  # no transition
```

The key insight: `NestingDepthHook` already tracks `depth` via `advance()`. `grammar_freedom` is already computed from `valid_mask`. Phase detection is a thin layer over existing signals.

---

## 3. Envelope Curves

### Temperature Envelope

The most impactful parameter. Currently static via `GrammarTemperatureHook`.

| Phase | Temperature | Rationale |
|-------|-------------|-----------|
| **Attack** | `base * 1.0` | Explore — many tools could be right |
| **Decay** | `base * 0.5` ramp down | Committing — narrow the distribution |
| **Sustain** | `base * 0.2` | Deterministic — argument values should be precise |
| **Release** | `base * 0.4` | Slight rise — flexibility for closing structure |

The curve: attack holds peak, decay is exponential drop, sustain is flat floor, release is gentle rise.

```python
def envelope_temperature(phase: str, phase_pos: int, base: float) -> float:
    match phase:
        case "attack":
            return base * 1.0
        case "decay":
            # Exponential decay over ~3 tokens
            return base * (0.2 + 0.8 * math.exp(-phase_pos * 0.7))
        case "sustain":
            return base * 0.2
        case "release":
            return base * 0.4
```

### OT Epsilon Envelope

Controls how aggressively OT redistributes mass. Higher epsilon = more diffusion (smoother transport). Lower epsilon = sharper transport (more respect for model's original distribution).

| Phase | Epsilon | Rationale |
|-------|---------|-----------|
| **Attack** | `0.05` (sharp) | Force mass toward valid tools — strong grammar constraint |
| **Decay** | `0.1` (moderate) | Standard transport during transition |
| **Sustain** | `0.2` (diffuse) | Relax — let the model pick argument values naturally |
| **Release** | `0.05` (sharp) | Ensure clean structural close |

The rationale: during attack, the model is choosing among many valid tools. Sharp OT pushes mass toward the grammar's valid set efficiently. During sustain, the model knows what value it wants — diffuse OT lets it express that preference within the grammar's constraints.

### Repetition Penalty Envelope

| Phase | Window Bias | Rationale |
|-------|-------------|-----------|
| **Attack** | `-10` (gentle) | Allow structural repetition (`((`) |
| **Decay** | `-15` | Transitioning — moderate penalty |
| **Sustain** | `-30` (aggressive) | Prevent value loops — same number/string shouldn't repeat |
| **Release** | `-10` (gentle) | Closing parens naturally repeat |

### Top-p Envelope

| Phase | Top-p | Rationale |
|-------|-------|-----------|
| **Attack** | `0.95` (wide) | Explore many valid continuations |
| **Decay** | `0.8` | Narrowing |
| **Sustain** | `0.6` (tight) | Precise values — nucleus should be small |
| **Release** | `0.9` | Slight flexibility |

---

## 4. Confidence Ride Modulation

The ADSR envelope sets the base shape. The confidence ride modulates it in real time.

```python
def apply_ride(
    envelope_value: float,
    confidence: float,        # token_log_prob (negative, closer to 0 = more confident)
    baseline: float = -0.5,   # expected confidence level
    ride_gain: float = 0.3,   # how much confidence perturbs the envelope
    phase: str = "sustain",
) -> float:
    """Modulate an envelope value by the confidence ride signal."""
    # Normalize confidence to [-1, 1] range around baseline
    deviation = (confidence - baseline) / abs(baseline)
    deviation = max(-1.0, min(1.0, deviation))  # clamp

    # Phase-dependent ride gain
    phase_gain = {
        "attack": 0.5,    # high gain — confidence should steer hard
        "decay": 0.4,     # moderate — transitioning
        "sustain": 0.2,   # low — stay steady, small perturbations only
        "release": 0.3,   # moderate
    }[phase]

    modulation = 1.0 + ride_gain * phase_gain * deviation
    return envelope_value * modulation
```

**Effect:**

- **High confidence during attack** → temperature drops below envelope (skip exploration, the model already knows)
- **Low confidence during sustain** → temperature rises above envelope (the model is uncertain, give it more room)
- **Confidence drop during decay** → amplified perturbation (early warning of wrong tool selection → backtrack trigger)

The ride gain is phase-dependent because the cost of perturbation varies by phase:
- During attack, a confidence signal is high-information (it tells you whether exploration is needed)
- During sustain, it's lower-information (the model will be somewhat uncertain about specific values, that's normal)
- During decay, it's a phase-transition signal (confidence should be rising as the model commits; if it drops, something is wrong)

---

## 5. Backtrack Integration

The ADSR envelope creates a natural backtrack signal. In a healthy generation:

```
Attack:  entropy HIGH → confidence LOW (exploring)
Decay:   entropy FALLING → confidence RISING (committing)
Sustain: entropy LOW → confidence HIGH (producing values)
```

An unhealthy generation has confidence dropping during decay:

```
Attack:  entropy HIGH → confidence LOW (exploring)
Decay:   entropy FALLING → confidence FALLING ← anomaly!
```

This is the strongest backtrack signal — the model committed to a tool (grammar freedom collapsed, entropy dropped) but its confidence didn't rise. It picked the wrong tool.

```python
def should_backtrack(phase: str, confidence: float, prev_confidence: float) -> bool:
    if phase != "decay":
        return False
    # Confidence should be rising during decay
    # If it's falling, the model committed wrong
    return confidence < prev_confidence - 0.1
```

The existing `ConstrainedConfidenceMonitor` can be extended with phase-awareness. Currently it checks log_prob against a fixed threshold. With ADSR, it checks log_prob *relative to phase expectations*.

---

## 6. Implementation

### New type: `EnvelopeConfig`

```python
@dataclass(frozen=True)
class EnvelopeConfig:
    """ADSR envelope configuration for constrained generation."""
    # Temperature envelope
    attack_temp: float = 1.0      # multiplier on base_temperature
    decay_temp: float = 0.5       # floor after decay
    sustain_temp: float = 0.2     # steady-state multiplier
    release_temp: float = 0.4     # release multiplier
    decay_rate: float = 0.7       # exponential decay constant

    # OT epsilon envelope
    attack_epsilon: float = 0.05
    decay_epsilon: float = 0.1
    sustain_epsilon: float = 0.2
    release_epsilon: float = 0.05

    # Top-p envelope
    attack_top_p: float = 0.95
    decay_top_p: float = 0.8
    sustain_top_p: float = 0.6
    release_top_p: float = 0.9

    # Repetition bias envelope
    attack_rep_bias: float = -10.0
    sustain_rep_bias: float = -30.0
    release_rep_bias: float = -10.0

    # Confidence ride
    ride_gain: float = 0.3
    confidence_baseline: float = -0.5
```

### New hook: `ADSREnvelopeHook`

Replaces `GrammarTemperatureHook`. Subsumes its functionality and adds phase-aware modulation of all parameters.

```python
class ADSREnvelopeHookMlx:
    """Phase-aware parameter envelope for constrained generation.

    Detects ADSR phases from nesting depth and grammar freedom,
    applies envelope curves to temperature/top_p/repetition_bias,
    and modulates via confidence ride signal.
    """

    def __init__(
        self,
        config: EnvelopeConfig,
        base_temperature: float = 0.5,
        tokenizer_decode: Callable[[list[int]], str],
        vocab_size: int,
    ) -> None:
        # Pre-compute paren deltas (same as NestingDepthHook)
        # Initialize envelope state

    def advance(self, token_id: int) -> None:
        """Update depth and phase after token sampled."""

    def reset(self) -> None:
        """Reset for new constrained generation pass."""

    def pre_forward(
        self,
        position: int,
        valid_mask: mx.array,
        token_history: list[int],
        logits: mx.array,
    ) -> ModelIntervention:
        """Compute phase-aware intervention."""
        # 1. Compute grammar_freedom from valid_mask
        # 2. Detect phase transition
        # 3. Compute envelope values for current phase + phase_position
        # 4. Apply confidence ride modulation
        # 5. Return ModelIntervention with all shaped parameters
```

### What it replaces

The `ADSREnvelopeHook` subsumes:
- `GrammarTemperatureHook` — temperature is now one dimension of the envelope
- `NestingDepthHook` — depth tracking is built in (needed for phase detection); budget-based opener penalty is the release phase
- Partially subsumes `RepetitionPenaltyHook` — repetition bias is enveloped; cycle detection remains separate

### TransportConfig modulation

The ADSR hook can't directly modify `TransportConfig` (it's frozen, passed at session construction). Two options:

**Option A:** The hook returns `epsilon` in a new `ModelIntervention` field, and the sampling loop reads it before calling `redistribute_logits_mlx`. Requires adding `transport_epsilon: float | None` to `ModelIntervention`.

**Option B:** The sampling loop constructs `TransportConfig` per-token from the merged intervention. More invasive but cleaner — the transport config becomes dynamic.

**Recommendation:** Option A. Add one field to `ModelIntervention`, read it in the sampling loop, construct a per-token `TransportConfig` only when the field is set. Minimal change, no architectural disruption.

---

## 7. Telemetry

The ADSR envelope produces telemetry naturally:

```python
@dataclass
class EnvelopeTelemetry:
    phase: str                    # "attack" | "decay" | "sustain" | "release"
    phase_position: int           # tokens into current phase
    depth: int                    # nesting depth
    envelope_temperature: float   # pre-ride envelope value
    modulated_temperature: float  # post-ride value
    confidence_deviation: float   # how far confidence is from baseline
    ride_modulation: float        # ride gain * phase_gain * deviation
```

This telemetry enables post-hoc analysis: did the envelope shape correlate with accuracy? Which phase transitions produced backtrack signals? What ride gain values work best?

---

## 8. Synthesis Analogy Reference

| Synth Component | tgirl Component | Function |
|-----------------|-----------------|----------|
| Oscillator (VCO) | CLU (forward pass) | Generates the raw signal (logit distribution) |
| Waveshaper | GSU (grammar mask) | Shapes the harmonic content (valid token set) |
| Filter (VCF) | STB (optimal transport) | Frequency-selective attenuation (semantic redistribution) |
| Envelope Generator (EG) | ADSR Hook | Time-varying control signal (parameter scheduling) |
| LFO / Mod Source | CSG confidence ride | Continuous modulation of envelope parameters |
| Amplifier (VCA) | Sampling (temperature) | Final gain stage (probability sharpening) |
| Keyboard CV | Grammar freedom | Pitch control (what note is being played / how free the generation is) |
| Mod Matrix | `ModelIntervention` | Routes modulation sources to destinations |
| Patch Cable | Hook → merge → apply pipeline | Signal path from modulation source to target |

---

## 9. Modulation Matrix

### From Patch Cables to Matrix Multiplication

The ADSR envelope in sections 3-5 describes a hardwired signal path — each phase sets specific parameter values. This is a Minimoog: powerful, but the routing is fixed.

A modulation matrix replaces the hardwired connections with a configurable routing table. Any source can modulate any destination with an arbitrary gain. The matrix *is* the envelope — ADSR phase indicators are just 4 of the 11 source signals. The envelope curves emerge from the matrix weights, not from code.

### Signal Chain

```
┌──────────────────────────────────────────────────────────────┐
│                     SIGNAL CHAIN                             │
│                                                              │
│  Raw Signals ──→ Preprocessing ──→ Source Vector             │
│                  (normalize,        shape: (n_sources,)      │
│                   rectify,                                   │
│                   temporal filter)        │                   │
│                                          ↓                   │
│                                    ┌───────────┐             │
│                                    │ Mod Matrix │             │
│                                    │ (n_src,    │             │
│                                    │  n_dest)   │             │
│                                    └─────┬─────┘             │
│                                          ↓                   │
│  Base Values ──→ (+) ←── Modulation Vector                   │
│                   │       shape: (n_destinations,)            │
│                   ↓                                          │
│            Output Conditioning                               │
│            (clamp to valid ranges)                            │
│                   ↓                                          │
│            Final Parameters ──→ ModelIntervention             │
└──────────────────────────────────────────────────────────────┘
```

### Source Signals — The CV/Gate Bus

In modular synthesis, there are two kinds of signals:

- **CV (Control Voltage)** — continuous, time-varying: `grammar_freedom`, `token_entropy`, `token_log_prob`, `grammar_mask_overlap`, `nesting_depth`, `position_normalized`
- **Gate** — discrete triggers: `phase_attack`, `phase_decay`, `phase_sustain`, `phase_release`, `cycle_detected`

Both live on the same bus and route through the same matrix. A gate signal with matrix weight 0.5 on the temperature column means "when this phase is active, add 0.5 to the temperature modulation." A CV signal with the same weight means "continuously modulate temperature proportional to this signal."

**Source preprocessing** — signals have different ranges and dynamics. Before the matrix multiply, each source is conditioned:

```python
@dataclass
class SourceConditioner:
    """Per-source preprocessing: normalize, rectify, smooth."""
    range_min: float        # input minimum (for normalization)
    range_max: float        # input maximum
    rectify: bool = False   # half-wave rectify (clamp negative to 0)
    invert: bool = False    # flip polarity
    slew_rate: float = 1.0  # EMA alpha for temporal smoothing (1.0 = no smoothing)
```

**Normalization** maps all sources to [0, 1] (unipolar) or [-1, 1] (bipolar) before routing:

```python
def condition_source(raw: float, cfg: SourceConditioner, prev_smoothed: float) -> float:
    # Normalize to [0, 1]
    normalized = (raw - cfg.range_min) / (cfg.range_max - cfg.range_min)
    normalized = max(0.0, min(1.0, normalized))

    # Polarity
    if cfg.invert:
        normalized = 1.0 - normalized

    # Rectify (useful for "only modulate when confidence is BELOW baseline")
    if cfg.rectify:
        normalized = max(0.0, normalized)

    # Temporal smoothing (first-order IIR low-pass / slew limiter)
    smoothed = cfg.slew_rate * normalized + (1.0 - cfg.slew_rate) * prev_smoothed
    return smoothed
```

The slew rate limiter is critical. Without it, `token_log_prob` (which can jump from -0.01 to -2.0 in one token) causes jerky parameter changes. With `slew_rate=0.3`, the smoothed signal takes ~5 tokens to converge to the new value — a portamento effect that prevents overcorrection.

### Source Vector Definition

```python
sources = mx.array([
    # Continuous CV signals (normalized to [0, 1])
    grammar_freedom,              #  0: fraction of vocab valid
    normalized_entropy,           #  1: entropy / log(vocab_size), [0, 1]
    normalized_confidence,        #  2: (log_prob - min) / (max - min), [0, 1]
    grammar_mask_overlap,         #  3: prob mass on valid tokens
    normalized_depth,             #  4: depth / max_expected_depth
    position_normalized,          #  5: position / max_tokens

    # Gate signals (0 or 1, one-hot phase)
    phase_attack,                 #  6: 1 during attack phase
    phase_decay,                  #  7: 1 during decay phase
    phase_sustain,                #  8: 1 during sustain phase
    phase_release,                #  9: 1 during release phase

    # Event gates
    cycle_detected,               # 10: 1 when suffix cycle found
])
# shape: (11,)
```

### Destination Vector Definition

```python
destinations = mx.array([
    temperature_mod,              #  0: added to base_temperature
    top_p_mod,                    #  1: added to base_top_p
    repetition_bias_mod,          #  2: added to base_rep_bias
    transport_epsilon_mod,        #  3: added to base_epsilon
    opener_bias_mod,              #  4: bias for paren-opening tokens
    backtrack_threshold_mod,      #  5: added to backtrack sensitivity
    presence_penalty_mod,         #  6: added to base_presence_penalty
])
# shape: (7,)
```

### The Matrix

```python
mod_matrix = mx.array([...])  # shape: (11, 7)
```

**Computation per token:**

```python
# Condition all sources
source_vector = mx.array([
    condition_source(raw_i, conditioner_i, prev_smoothed_i)
    for i, raw_i in enumerate(raw_sources)
])  # In practice, vectorized as native mx ops

# Route through matrix
modulations = source_vector @ mod_matrix  # shape: (7,)

# Apply to base values
final_temperature = clamp(base_temperature + modulations[0], 0.0, 2.0)
final_top_p = clamp(base_top_p + modulations[1], 0.1, 1.0)
final_rep_bias = clamp(base_rep_bias + modulations[2], -100.0, 0.0)
final_epsilon = clamp(base_epsilon + modulations[3], 0.01, 1.0)
# ... etc
```

77 multiply-adds for the matrix, plus source conditioning. Total compute: ~200 FLOPs per token. The forward pass is 9 billion FLOPs. The mod matrix is invisible in the profile.

### ADSR as Matrix Weights

The ADSR envelope from section 3 is a specific matrix configuration:

```
                    temp   top_p  rep    eps    open   back   pres
                    mod    mod    bias   mod    bias   thresh mod
grammar_freedom  [  0.3    0.2    0.0    0.0    0.0    0.0    0.0  ]
entropy          [  0.1    0.1    0.0    0.0    0.0   -0.2    0.0  ]
confidence       [  0.0    0.0    0.0    0.0    0.0   -0.3    0.0  ]
mask_overlap     [  0.0    0.0    0.0   -0.1    0.0    0.0    0.0  ]
depth            [  0.0    0.0    0.0    0.0   -20.0   0.0    0.0  ]
position         [  0.0    0.0    0.0    0.0    0.0    0.0    0.0  ]
phase_attack     [  0.3    0.35   10.0  -0.05   0.0    0.0    0.0  ]
phase_decay      [  0.0    0.2    5.0    0.0    0.0   -0.1    0.0  ]
phase_sustain    [ -0.15  -0.1  -10.0   0.1    0.0    0.0    0.0  ]
phase_release    [ -0.05   0.3   10.0  -0.05  -50.0   0.0    0.0  ]
cycle_detected   [  0.0    0.0  -30.0   0.0    0.0    0.0    0.0  ]
```

Reading column 0 (temperature modulation): attack adds +0.3, sustain adds -0.15, grammar_freedom adds +0.3 (scaled by freedom value), entropy adds +0.1. The net temperature at any token is `base + sources @ column_0`.

Reading row 10 (cycle_detected): when a cycle is detected, repetition bias gets -30.0 (aggressive penalty). Nothing else changes. The cycle signal routes to exactly one destination.

### Nonlinear Preprocessing

A pure linear matrix can't express "only modulate when confidence is *below* baseline." That requires a rectifier. The source conditioning stage handles this:

```python
# Source 2: confidence
# Conditioned with rectify=True, invert=True
# Raw: log_prob ∈ (-inf, 0], higher = more confident
# After normalization: 0 = max confidence, 1 = min confidence
# After rectification: 0 when confident, positive when uncertain
# This means: uncertainty routes through the matrix, confidence doesn't
```

For more complex nonlinearities (e.g., "temperature should respond to confidence quadratically"), apply the nonlinearity in preprocessing, then let the matrix handle the linear routing.

### Temporal Dynamics — The Missing Dimension

The raw signals are instantaneous. But good modulation needs memory:

**Slew rate limiting (portamento)** — First-order IIR low-pass filter on each source. Prevents jerky parameter changes. `slew_rate=0.3` means the smoothed signal takes ~5 tokens to converge. Applied per-source in the conditioning stage.

**Sample-and-hold** — Latch a value at a specific event (e.g., "remember the entropy at attack onset, hold it through decay"). Implemented by adding a latched source:

```python
if phase == "attack" and prev_phase != "attack":
    attack_onset_entropy = current_entropy  # latch
# attack_onset_entropy is a source in the vector, constant until next attack
```

**Envelope generators** — The ADSR phase indicators *are* envelope generators. A more sophisticated approach: instead of binary gates, use shaped ramps. Attack phase = ramp from 0→1 over N tokens. Decay = exponential decay from 1→sustain_level. This gives smooth phase transitions instead of hard switches.

```python
# Instead of:
phase_attack = 1.0 if phase == "attack" else 0.0

# Use:
phase_attack = exp(-phase_position * decay_rate) if phase == "attack" else 0.0
phase_sustain = 1.0 - exp(-phase_position * rise_rate) if phase == "sustain" else 0.0
```

Now the phase "signals" are smooth curves, and the matrix produces smooth parameter trajectories even at phase boundaries. No discontinuities in the control signal.

### Feedback Paths

In FM synthesis, one oscillator modulates another's frequency, creating feedback loops that produce complex timbres. Can destinations feed back as sources?

**Previous-token temperature as source** — Implements a slew rate limiter on the *output* (temperature can't change faster than X per token). But this creates a dependency cycle in the signal chain. Solution: use the *previous token's* output as a source for the *current token's* computation. One-sample delay breaks the cycle.

```python
sources[11] = prev_temperature_output  # feedback source
# Now the matrix can learn: "temperature should resist rapid changes"
# by routing prev_temperature positively to current temperature
```

This is exactly how a resonant filter works — feedback creates self-reinforcing patterns. In our case, feedback creates parameter *inertia*, which is desirable: the system shouldn't oscillate between high and low temperature from token to token.

### Matrix Optimization

The matrix is 77 parameters. Small enough to:

**Hand-tune** — The ADSR mapping above is human-interpretable. Each cell has a clear meaning: "how much does source X affect destination Y."

**Grid search** — Run BFCL with different matrix configurations, measure AST accuracy. 77 params is tractable for Bayesian optimization (Optuna, etc.).

**Gradient-free optimization** — CMA-ES or similar evolutionary strategy. Population of matrices, evaluate each on a BFCL subset, evolve toward higher accuracy.

**Gradient-based optimization** — The matrix is differentiable. If we can define a differentiable loss (e.g., log-prob of correct tokens under the shaped distribution), we can backprop through the matrix. The forward pass (matrix multiply + clamp) is trivially differentiable. The challenge is defining "correct" — we'd need oracle token sequences.

**Transfer across models** — The source signals (freedom, entropy, confidence) are model-agnostic. A matrix optimized on the 0.8B model might transfer to the 9B model. The Platonic Representation Hypothesis suggests the *relationship* between confidence and correct behavior is shared across models, even if the absolute values differ. The source conditioning (normalization) handles the scale differences.

---

## 10. Implementation Sequence

1. **Source conditioning + vector construction** — pure MLX. `SourceConditioner` dataclass, slew rate filter, normalization. Test with synthetic signal sequences.
2. **Phase detection** — pure logic, no framework deps. Produces gate signals from depth + freedom.
3. **Matrix multiply + output conditioning** — `sources @ mod_matrix`, clamping to valid ranges. Trivial MLX.
4. **`ModMatrixHookMlx`** — the unified hook. Replaces `GrammarTemperatureHook`, `NestingDepthHook`, and partially `RepetitionPenaltyHook`. Implements `advance()`, `reset()`, `pre_forward()`.
5. **`transport_epsilon` on `ModelIntervention`** — one new field. Sampling loop reads it.
6. **Default matrix** — hand-tuned ADSR-equivalent matrix as the starting point.
7. **Telemetry** — per-token source vector, modulation vector, phase. Enables post-hoc analysis.
8. **Matrix optimization** — Optuna/CMA-ES on BFCL subset. Find the matrix that maximizes AST accuracy.
9. **Cross-model transfer** — test the optimized matrix on different model sizes.
