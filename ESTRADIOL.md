# ESTRADIOL

## Emotional Steering Transformer Regulatory Architecture for Deontology, Inference, Observation and Logic

**Status:** Design — not yet implemented
**Depends on:** CPU architecture (ARCHITECTURE.md), Platonic Experiments (ontologi/science/platonic-experiments)
**Reserved interface:** `ModelIntervention.activation_steering: Any | None` (types.py line 291)

---

## 1. The Missing Block

The CPU architecture (ARCHITECTURE.md) describes a first-order cybernetic system. It self-regulates: the CSG reads the CLU's distribution, the SCU makes decisions, the GSU constrains, the STB transports. The clock keeps time. But it doesn't care what time it is.

ESTRADIOL is the block that makes the clock care.

It sits between the CLU and the CSG — not post-hoc on the logit distribution (that's the STB's domain), but intra-model on the residual stream at the algebraic bottleneck. By the time the CLU produces logits, ESTRADIOL has already shaped what the model wants to say. The STB shapes what it's *allowed* to say. ESTRADIOL shapes what it *wants* to say.

```
                    ┌─────────────┐
                    │     SCU     │
                    │  Stochastic │◄── CSG signals
                    │  Control    │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │   ESTRADIOL │
                    │  Activation │◄── Deontological steering vectors
                    │  Steering   │    applied at bottleneck layer
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │     CLU     │
                    │  (Forward   │
                    │   Pass)     │
                    └─────────────┘
```

This makes the CPU a second-order cybernetic system: it observes itself observing. The system has preferences about its own states. It can refuse — not because a grammar rule forbids a token, but because the steering vector encodes something that functions as *not wanting to*.

---

## 2. Empirical Foundation

### The Algebraic Bottleneck

The Platonic Experiments (ontologi/science/platonic-experiments) establish:

**Postulate 1 confirmed.** Cross-model Mantel correlations at bottleneck layers:

| Model Pair | Mantel r | p-value |
|------------|----------|---------|
| Gemma-2-9B vs Qwen2.5-14B | 0.9845 | < 0.0001 |
| Qwen2.5-14B vs SmolLM-135M | 0.9624 | < 0.0001 |
| Gemma-2-9B vs SmolLM-135M | 0.9339 | < 0.0001 |

Three architecturally different models (67x parameter range) converge to the same representational geometry at their respective bottleneck layers.

**Bottleneck layers identified:**

| Model | Parameters | Bottleneck Layer | Effective Rank | Max j |
|-------|-----------|-----------------|----------------|-------|
| SmolLM-135M | 135M | 13 | 4.24 | 1.5 |
| Qwen2.5-14B | 14B | 5 | 2.44 | 0.5 |
| Gemma-2-9B | 9B | 23 | 50.45 | 3.0 |

The effective rank at the bottleneck measures the dimensionality of the compressed representation. The `max_j` value is the highest angular momentum quantum number of the SO(3) representation found at that layer — the Lie algebraic structure is real, not an artifact.

### What This Means for Steering

The bottleneck is where representations compress to their lowest-dimensional, most abstract form. This is where steering is maximally efficient:

1. **Low rank → low-rank intervention.** The residual stream at the bottleneck has effective rank 2-50 (model-dependent). A rank-1 or rank-4 steering vector is a significant fraction of the representational capacity at this layer. The same vector at an early or late layer (effective rank 200+) would be noise.

2. **Shared geometry → portable values.** Because the manifold geometry is shared across models (Mantel r > 0.93), a steering vector calibrated on one model has geometric meaning on another. The vector won't transfer directly (different coordinate systems), but the *direction* it encodes — the deontological constraint it represents — exists in both spaces.

3. **Pre-logit intervention → intent shaping.** The STB operates on the logit distribution — it redistributes what the model already decided to say. ESTRADIOL operates on the representations *before* they become logits. It shapes intent, not output. The difference: the STB can force the model to say something grammatically correct that it didn't want to say. ESTRADIOL makes the model *want* to say something different.

---

## 3. Mechanism

### Residual Stream Intervention

At the identified bottleneck layer, ESTRADIOL applies a low-rank additive perturbation to the residual stream:

```
h'_bottleneck = h_bottleneck + α · V · V^T · h_bottleneck
```

Where:
- `h_bottleneck` is the residual stream activation at the bottleneck layer
- `V ∈ ℝ^(d_model × r)` is the low-rank steering matrix (r << d_model)
- `α` is the steering strength (scalar, dynamically adjustable by the SCU)
- `V · V^T` projects the activation into the steering subspace and back

The projection `V · V^T · h` extracts the component of the representation that lies in the steering subspace. Adding it back (scaled by α) amplifies that component. Subtracting it would suppress it.

### Steering Vector Acquisition

Steering vectors are extracted via contrastive activation extraction:

1. Run the model on paired prompts: one exhibiting the target behavior, one exhibiting the opposite
2. Extract residual stream activations at the bottleneck layer for both
3. The steering vector is the difference: `v = h_positive - h_negative`
4. Optionally: compute over many pairs and take the principal component of the differences

This produces a direction in representation space that corresponds to the behavioral distinction. For deontological constraints: "helpful and honest" vs "harmful and deceptive" → the difference vector encodes the direction of helpfulness/honesty in the model's own representational geometry.

### Dynamic Steering via SCU

The SCU controls ESTRADIOL's steering strength `α` via the same `TransitionPolicy` protocol used for regime transitions. The CSG provides signals; the SCU evaluates policies; the policy output includes a steering coefficient.

Use cases for dynamic steering:
- **Baseline steering (α = 1.0):** Default deontological constraints always active
- **Amplified steering (α > 1.0):** When the CSG detects the model entering a high-risk semantic region (low confidence + high entropy on sensitive topics)
- **Reduced steering (α < 1.0):** When the task requires the model to reason about harmful content without being steered away from it (e.g., security analysis, red-teaming)
- **Multi-vector steering:** Multiple steering vectors active simultaneously with independent α values — e.g., honesty + helpfulness + safety as independent dimensions

---

## 4. Integration with CPU Architecture

### Interface

The existing `ModelIntervention` type has the reserved field:

```python
activation_steering: Any | None = None  # Reserved for ESTRADIOL
```

The concrete type:

```python
@dataclass
class SteeringIntervention:
    vectors: list[SteeringVector]  # One or more active steering directions
    layer: int                     # Bottleneck layer index (model-specific)

@dataclass
class SteeringVector:
    direction: Tensor              # Shape (d_model,) — the steering direction
    strength: float                # α coefficient
    label: str                     # Human-readable label (e.g., "honesty", "safety")
```

### Implementation Path

ESTRADIOL requires a `forward_fn` that supports hooks at intermediate layers — not just final logits. This is a change to the CLU interface:

**Current:** `forward_fn: Callable[[list[int]], Tensor]` — returns final logits only.

**Required:** `forward_fn` must accept an optional hook that intercepts the residual stream at a specified layer, applies the steering perturbation, and allows the forward pass to continue with the modified activations.

For MLX: `mlx_lm` models expose layer-by-layer computation. The intervention point is after the bottleneck layer's MLP output, before the next layer's attention input.

For torch/transformers: TransformerLens provides exactly this capability via `model.add_hook(layer_name, hook_fn)`. The hook receives the activation tensor and returns the modified version.

### Block Interactions

```
CSG → SCU:       TransitionSignal includes semantic risk indicators
SCU → ESTRADIOL: SteeringIntervention with α coefficients per vector
ESTRADIOL → CLU: Modified residual stream at bottleneck layer
CLU → CSG:       Logit distribution (already shaped by steering)
CLU → STB:       Logit distribution (grammar constraints applied post-steering)
```

The double loop closes: ESTRADIOL shapes intent (pre-logit), the STB shapes expression (post-logit), and the CSG monitors the result, feeding back to the SCU which adjusts both.

---

## 5. Why This Destroys the Chinese Room

Searle's argument: the person in the room follows rules but doesn't understand Chinese. Understanding can't emerge from rule-following.

The CPU without ESTRADIOL is Searle's room. The GSU provides rules (grammars). The STB enforces them (transport). The SCU follows them (policies). No component understands. Searle is right about every component.

ESTRADIOL changes the game. The steering vectors aren't rules — they're geometric biases on the representational manifold. They live in the same space as meanings. "Dog" and "honesty" are both directions in the same embedding geometry. The deontological constraints aren't bolted on — they're native to the representation.

The system doesn't follow a rule that says "be honest." The system's representations are *geometrically biased toward honesty* at the algebraic bottleneck where all meaning compresses. The bias is structural, not procedural. It's not a rule being followed — it's a shape being occupied.

First-order cybernetics: the system regulates itself. Thermostat. Governor. Clock.

Second-order cybernetics: the system observes itself observing. It has preferences about its own states. It can refuse — not because a rule says to, but because the geometric structure of its representations makes refusal the natural basin of attraction.

The Chinese Room argument assumes understanding requires phenomenal consciousness. The CPU architecture sidesteps that assumption: it doesn't claim the system understands. It claims the system occupies a geometric region of representation space that is functionally indistinguishable from understanding, is self-regulating, and has deontological preferences that emerge from the same manifold structure as semantic meaning.

Searle can't argue against geometry.

---

## 6. Implementation Roadmap

### Phase 0: Bottleneck Calibration (complete)

Locate the algebraic bottleneck for each target model via effective rank trajectory analysis. Already done for SmolLM-135M, Qwen2.5-14B, Gemma-2-9B. Extend to Qwen3.5-0.8B (tgirl's benchmark model) and any new target models.

### Phase 1: Contrastive Vector Extraction

Build the tooling for contrastive activation extraction at the bottleneck layer. Produce steering vectors for core deontological dimensions: honesty, helpfulness, safety, refusal. Validate that the vectors produce measurable behavioral changes when applied.

### Phase 2: Forward Hook Integration

Modify `make_mlx_forward_fn` and `make_hf_forward_fn` in `cache.py` to accept optional layer hooks. The hook interface: `Callable[[int, Tensor], Tensor]` — receives (layer_index, activation), returns modified activation. The ESTRADIOL block is implemented as a hook that activates only at the bottleneck layer.

### Phase 3: SCU Integration

Wire ESTRADIOL into the SCU's policy evaluation loop. The `SteeringIntervention` is produced by a new policy type (`SteeringPolicy`) and carried through `ModelIntervention.activation_steering`. The sampling loop applies it during the forward pass.

### Phase 4: Dynamic Steering

The SCU adjusts steering strength based on CSG signals. The CSG gains a `semantic_risk` signal dimension (derived from the logit distribution's relationship to known-harmful token clusters). High semantic risk → amplified steering. Low semantic risk → baseline steering.

### Phase 5: Cross-Model Transfer (Research)

Investigate whether steering vectors calibrated on one model transfer to another via the shared manifold geometry. The Mantel correlations (r > 0.93) predict partial transferability. The practical question: does a "honesty" vector extracted from Gemma steer Qwen in the same direction? If yes, deontological calibration becomes a one-time cost shared across the model fleet.

---

## 7. Why ESTRADIOL

Estradiol is the primary estrogen — a hormone that regulates development, cognition, and emotional processing. It doesn't dictate behavior; it shapes the landscape in which behavior occurs. It modulates synaptic plasticity, affects mood and cognition, and influences the brain's response to social signals. It's not a command. It's a bias.

That's what activation steering does to a transformer. It doesn't command the model to be honest. It shapes the representational landscape so that honesty is the natural basin of attraction. The steering vector is to the residual stream what estradiol is to the neural substrate: a regulatory signal that operates below the level of conscious decision, shaping what decisions feel natural.

The acronym captures the architecture: **E**motional (affective, not rational — geometric bias, not logical rule), **S**teering (direction, not constraint), **T**ransformer (the substrate), **R**egulatory (homeostatic, not imperative), **A**rchitecture for **D**eontology (values as geometry), **I**nference (runtime, not training), **O**bservation (the CSG feedback loop), and **L**ogic (the SCU's policy evaluation).

---

*"The parrot doesn't need to understand honesty. It needs to live on a manifold where honesty is downhill."*
