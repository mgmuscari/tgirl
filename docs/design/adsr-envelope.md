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

## 9. Implementation Sequence

1. **`EnvelopeState` + phase detection** — pure logic, no framework deps. Test with unit tests on synthetic sequences.
2. **`envelope_*` curve functions** — pure math. Test that each produces expected values at phase boundaries.
3. **`apply_ride` modulation** — pure math. Test that confidence perturbation scales correctly.
4. **`ADSREnvelopeHookMlx`** — integrates phase detection, curves, and ride. Replaces `GrammarTemperatureHook` and `NestingDepthHook`.
5. **`transport_epsilon` on `ModelIntervention`** — one new field. Sampling loop reads it.
6. **Telemetry integration** — add `EnvelopeTelemetry` to per-token recording.
7. **Benchmarking** — BFCL before/after comparison. The ADSR hypothesis: phase-aware parameters improve accuracy on entries where the 0.8B model fails due to wrong argument values (sustain phase improvements) or wrong tool selection (attack/decay phase improvements).
