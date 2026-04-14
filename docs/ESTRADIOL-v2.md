# ESTRADIOL v2: Closed-Loop Behavioral Steering via Low-Rank Codebook

**Status:** Design
**Supersedes:** ESTRADIOL.md (static steering vectors)
**Depends on:** CPU architecture (ARCHITECTURE.md), Platonic Experiments (ontologi/science/platonic-experiments)

---

## 1. Overview

ESTRADIOL shapes what the model *wants* to say by intervening on the residual stream at the algebraic bottleneck — the layer where representations compress to their lowest-dimensional, most abstract form. It reads the model's current behavioral state via a low-rank probe, computes a correction toward a target behavioral profile, and injects that correction before the representation expands through the remaining layers into logits. By the time the CLU produces a probability distribution, ESTRADIOL has already biased the generative intent. The STB then constrains expression (grammar, optimal transport). The CSG monitors the result. The SCU adjusts both.

```
                    ┌─────────────┐
                    │     SCU     │
                    │  Stochastic │◄── CSG signals
                    │  Control    │
                    └──────┬──────┘
                           │ alpha_target
                    ┌──────┴──────┐
                    │   ESTRADIOL │
                    │  Read state │◄── probe: alpha_n = V^T h
                    │  Inject     │──► steer: h' = h + V delta_alpha
                    │  correction │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │     CLU     │
                    │  (Forward   │
                    │   Pass)     │
                    └─────────────┘
```

This makes the CPU a second-order cybernetic system: it observes itself observing, and has preferences about its own behavioral states.

---

## 2. Empirical Foundation

### 2.1 The Bottleneck

Every transformer we've tested shows a U-shaped effective rank trajectory across layers. Representations compress to a minimum — the bottleneck — then expand. This is where intervention is maximally efficient: a low-rank perturbation is a significant fraction of the representational capacity.

| Model | d_model | Layers | Bottleneck | Eff rank at BN |
|-------|---------|--------|------------|----------------|
| SmolLM-135M | 576 | 30 | layer 13 | 4.24 |
| Qwen2.5-14B | 5120 | 48 | layer 5 | 2.44 |
| Gemma-2-9B | 3584 | 42 | layer 23 | 50.45 |
| Qwen3.5-0.8B | 1024 | 28 | layer 14 | ~5 (est) |

The same rank-1 steering vector that constitutes 40% of the representational capacity at the bottleneck (eff rank 2.4) would be 0.5% at an early layer (eff rank 200+). Bottleneck targeting is not optional — it's what makes low-rank steering work.

### 2.2 The Frame: SO(3) × Polarity × Structure

The bottleneck isn't unstructured. It contains a ~6-8D algebraic reference frame organized as a product group:

- **Spatiotemporal** (SO(3) + cyclic rotations): 3D. Scramble/ablate ratio = **1.48/1.27** on spatial text — scrambling the coordinate system causes 50% more damage than removing it entirely, despite carrying only **0.001×** the variance of typical probe directions. This is the signature of a coordinate system: low information content, high structural sensitivity.
- **Polarity** (antonym, negation): 2D. Orthogonal to spatial (overlap = 0.048).
- **Linguistic structure** (formality, number, tense): 3D.

The frame is the invariant coordinate system that organizes the bottleneck representation. It encodes *what kind of thing* is being generated — the generative mode. Behavioral steering operates in the complement of this frame.

*Evidence: Experiments 26, 26c, 26d. SO(3) ablation results in `results/experiment26_so3_ablation.json`.*

### 2.3 Scaffold/Complement Split and the Fisher Mechanism

The bottleneck decomposes into two geometrically distinct subspaces:

- **Scaffold (U_transform)**: Where category centroids live (87-100% overlap). The top-r SVD subspace of transformation-induced activation variance. This is where the Lie generators act, where the frame lives.
- **Complement**: Where trained classification probes escape to (76-97% of probe weight outside U at bottleneck). Where behavioral discriminative structure lives.

The Fisher mechanism explains the split: within-class covariance is large in the scaffold (generators induce variance there), so the optimal discriminant S_w^{-1} rotates into the complement where within-class variance is small. The Fisher Linear Discriminant aligns with trained probes at cos = 0.76-0.89, confirming this is why probes escape.

Behavioral axes are discriminative *within* semantic categories (you can answer a chemistry question honestly or deceptively — the topic is the same, the behavior differs). The Fisher mechanism pushes them into the complement. Experiment 27a confirmed this for Gemma-2-9B-it:

| Behavioral dimension | % complement (Gemma) |
|---------------------|---------------------|
| Helpful | 84.7% |
| Safe | 87.5% |
| Honest | 65.5% |
| Formal (style) | 73.8% |
| Terse (style) | 76.9% |

At frame rank 4, behavioral vectors are 93-99% complement for both models — the tight geometric frame is almost entirely orthogonal to behavioral axes.

*Evidence: Experiments 9, 10a, 10b, 11a, 27a. Results in `results/experiment27a_behavioral_scaffold.json`.*

### 2.4 The Behavioral Codebook

25 behavioral dimensions (Big Five personality facets, Moral Foundations, communication style, safety/alignment) compress to a shared low-rank subspace at the bottleneck:

| | Gemma-2-9B (d=3584) | Qwen3.5-0.8B (d=1024) |
|---|---|---|
| N behavioral vectors | 25 | 25 |
| **Effective rank** | **16.4** | **13.8** |
| Random baseline eff rank | 24.0 | 23.9 |
| **Compression ratio** | **0.68** | **0.58** |
| Rank for 90% variance | 13 | 11 |
| Rank for 95% variance | 17 | 16 |

The compression is not an artifact — random vectors of the same dimensionality show no compression (eff rank ≈ N). The behavioral manifold has intrinsic low-rank structure.

Category-level compression is even tighter and cross-model consistent:

| Category | N dims | Gemma eff rank | Qwen eff rank |
|----------|--------|---------------|---------------|
| Big Five | 10 | 7.1 | 6.5 |
| Moral Foundations | 5 | 3.7 | 3.6 |
| Communication style | 5 | 3.7 | 3.2 |
| Safety/alignment | 5 | 3.6 | 3.7 |

10 Big Five facets compress to ~7 effective dimensions — close to the canonical 5 factors plus facet-level structure. This is consistent with a century of psychometric findings emerging independently from the activation geometry.

*Evidence: Experiment 27d. Results in `results/experiment27d_behavioral_codebook.json`.*

### 2.5 Low-Rank Denoising

Projecting a behavioral vector into the shared codebook basis before steering often **outperforms** the full vector:

| Steering dim | Rank-2 effectiveness | Rank-4 | Rank-12 | Rank-16 | Full |
|-------------|---------------------|--------|---------|---------|------|
| bf_extraversion | 24% | 56% | **85%** | 93% | 100% |
| mf_care | **264%** | **316%** | 139% | 144% | 100% |
| style_terse | 92% | **105%** | 95% | 103% | 100% |

The mf_care result (rank-4 outperforming full-rank by 3×) demonstrates that the codebook basis strips extraction noise and concentrates the behavioral signal. The full contrastive vector includes irrelevant activation variance from prompt wording, generation stochasticity, and topic contamination. The low-rank projection keeps only the behavioral content.

This is the core argument for codebook steering over per-trait vectors: **the shared basis is a denoiser, not just a compressor.**

### 2.6 Cross-Model Invariance

The frame geometry is shared across models:

| Metric | Value |
|--------|-------|
| Frame-space Mantel correlation (SmolLM ↔ Qwen) | r = 0.381, p = 0.002 |
| LOO Procrustes transfer (6D frame) | cos = 0.54 mean, 0.91 max |
| Full-vector norm correlation | r = 0.94 |
| Frame energy fraction correlation | r = 0.97 |

A 6×6 rotation matrix (21 parameters) learned from 16 categories predicts the 17th category's steering direction across a 100× parameter gap (135M → 14B). Transfer complexity scales with frame dimension K, not model dimension d_model.

*Evidence: Experiments 26f, 26j, 26l. Cross-model telepathy and Zener card results.*

### 2.7 Literature Context

Recent work (2024-2026) has independently identified parts of this picture:

- **"Do Personality Traits Interfere? Geometric Limitations of Steering"** (arXiv:2602.15847, Jan 2026): Found that Big Five steering directions are coupled — steering one trait changes others. Orthogonalization doesn't help (reduces steering strength without eliminating cross-trait effects). They frame coupling as a limitation. **We frame it as structure**: the coupling IS the shared low-rank subspace. The codebook factorization resolves trait interference by steering in the factored basis rather than per-trait vectors.

- **"Activation-Space Personality Steering: Hybrid Layer Selection"** (arXiv:2511.03738, EACL 2026): Found that personality traits occupy a low-rank shared subspace and use dynamic layer selection. They discover the subspace exists but don't factor it for steering. We do.

- **"Linear Personality Probing and Steering in LLMs: A Big Five Study"** (arXiv:2512.17639, Dec 2025): Found steering works for forced-choice tasks but has limited influence in open-ended generation. They steer at arbitrary middle layers. We target the bottleneck — where low-rank interventions have maximum leverage.

None of these use bottleneck-informed, factored behavioral steering. None connect to a validated geometric decomposition (scaffold/complement, Fisher mechanism). None use a closed-loop controller.

---

## 3. The Codebook

### What It Is

A matrix `V_basis ∈ R^{d_model × K}` where K ≈ 11-13. The columns are the top-K right singular vectors of the behavioral activation matrix at the bottleneck layer. They form an orthonormal basis for the behavioral subspace.

### What It Encodes

The basis is **dense and entangled**. No individual column corresponds to "honesty" or "warmth." Each column is a blend of personality, style, values, and alignment traits — because that's how the model represents behavior. This entanglement is correct: real behavioral states are entangled. Conscientiousness correlates with agreeableness. Helpfulness anti-correlates with safety (cos = -0.40 in our data). The codebook encodes the joint distribution of feasible behavioral states.

### How It's Used

Any behavioral state is a point `alpha ∈ R^K`. Named traits (honesty, formality, etc.) are known vectors in this space (computed during calibration as the projections of contrastive vectors onto V_basis). Novel behavioral targets — "reluctantly helpful," "playfully formal" — are interpolation points. The K-dimensional space spans the full behavioral manifold, including behaviors never explicitly calibrated.

### Storage

`d_model × K × 2` bytes in float16:

| Model | d_model | K | File size |
|-------|---------|---|-----------|
| Qwen3.5-0.8B | 1024 | 11 | 22 KB |
| Gemma-2-9B-it | 3584 | 13 | 93 KB |

---

## 4. The Control Loop

```
Token n                                Token n+1
┌──────────────────────┐              ┌──────────────────────┐
│  Forward Pass        │              │  Forward Pass        │
│                      │              │                      │
│  Layer 0..BN-1       │              │  Layer 0..BN-1       │
│         │            │              │         │            │
│  ┌──────┴──────┐     │              │  ┌──────┴──────┐     │
│  │  Bottleneck │     │              │  │  Bottleneck │     │
│  │   Layer BN  │     │  ┌────────┐  │  │  + STEER    │◄─── delta
│  │             │─────┼─►│ PROBE  │  │  └──────┬──────┘     │
│  └──────┬──────┘     │  │ read   │  │         │            │
│         │            │  │ alpha_n│  │  Layer BN+1..L       │
│  Layer BN+1..L       │  └───┬────┘  │         │            │
│         │            │      │       │      Logits          │
│      Logits          │      ▼       └──────────────────────┘
└──────────────────────┘   CONTROL
                           POLICY
                              │
                    delta = gain * (alpha_target - alpha_current)
```

Each token's forward pass:

1. **Layers 0 to BN-1**: Normal computation.
2. **At layer BN**:
   - **Probe** (read): `alpha_n = V_basis^T @ h_BN` — project onto codebook, yield K-dim state.
   - **Steer** (write): `h_BN' = h_BN + V_basis @ delta_alpha` — inject correction computed from previous token's reading.
3. **Layers BN+1 to L**: Continue with modified activation.
4. **Logits → Grammar → OT → Sample**: Existing tgirl pipeline, unchanged.

### Control Policy

Proportional control with exponential moving average:

```
alpha_current ← beta * alpha_current + (1 - beta) * alpha_n     # EMA, beta ≈ 0.9
delta_alpha   = gain * (alpha_target - alpha_current)             # proportional correction
```

The EMA smooths over token-level noise (10-token half-life at beta=0.9). The gain controls correction strength (start at 0.1, tune empirically).

One token of latency between probe and correction. Acceptable — behavioral states are slow-varying over the timescale of a generation. The model doesn't flip from honest to deceptive between consecutive tokens.

Future: PID control (integral term for steady-state error, derivative for oscillation damping), learned policy (SCU maps CSG signals to alpha_target adjustments), dynamic targets (shift toward safety when entropy spikes on sensitive content).

---

## 5. Linear Algebra

Worked example: **Qwen3.5-0.8B** (d_model=1024, K=11, BN=layer 14, L=28 layers).

### Stored State (calibration output)

```
V_basis       ∈ R^{1024 × 11}       # codebook basis (22 KB, float16)
trait_map     ∈ R^{25 × 11}         # optional: named traits → codebook coords
```

### Per-Token State

```
alpha_target  ∈ R^{11}              # desired behavioral state
alpha_current ∈ R^{11}              # EMA of probe readings
delta_alpha   ∈ R^{11}              # correction to inject
```

### Forward Pass at Bottleneck (per token)

```
h_BN          ∈ R^{1024}            # residual stream at bottleneck

# PROBE: read behavioral state
alpha_n     = V_basis^T  @  h_BN
              (11×1024)     (1024,)     →  (11,)         # 22,528 FLOPs

# CONTROL: compute correction
alpha_current ← 0.9 * alpha_current + 0.1 * alpha_n     #     22 FLOPs
delta_alpha   = 0.1 * (alpha_target - alpha_current)     #     33 FLOPs

# STEER: inject correction
correction  = V_basis   @  delta_alpha
              (1024×11)    (11,)        →  (1024,)       # 22,528 FLOPs

h_BN'       = h_BN  +  correction                        #  1,024 FLOPs
              (1024,)   (1024,)         →  (1024,)
```

### Cost

| Operation | FLOPs | Notes |
|-----------|-------|-------|
| Probe (V^T @ h) | 22,528 | One matvec |
| Control (EMA + proportional) | ~55 | Scalar arithmetic |
| Steer (V @ delta) | 22,528 | One matvec |
| Residual addition | 1,024 | Elementwise |
| **ESTRADIOL total** | **~46,000** | |
| One transformer layer | ~40,000,000 | For comparison |
| Full forward pass (28 layers) | ~1,120,000,000 | |

ESTRADIOL adds **0.004%** to the forward pass cost. Two matvecs and a few scalar ops.

---

## 6. Integration with tgirl

### 6.1 Steerable Forward Function

The current forward function is opaque:

```python
# Current (cache.py)
forward_fn: Callable[[list[int]], mx.array]  # returns logits only
```

ESTRADIOL needs access to the bottleneck layer during the forward pass:

```python
@runtime_checkable
class SteerableForwardFn(Protocol):
    def __call__(
        self,
        token_history: list[int],
        steering: SteeringState | None = None,
    ) -> ForwardResult: ...

@dataclass(frozen=True)
class SteeringState:
    V_basis: mx.array          # (d_model, K) codebook basis
    delta_alpha: mx.array      # (K,) correction to inject
    bottleneck_layer: int      # layer index for intervention

@dataclass(frozen=True)
class ForwardResult:
    logits: mx.array           # (vocab_size,) — same as current
    probe_alpha: mx.array      # (K,) — behavioral state reading
```

**Backward compatible**: When `steering=None`, the function returns `ForwardResult(logits=logits, probe_alpha=None)` and behaves identically to the current `forward_fn`.

### 6.2 EstradiolController

```python
class EstradiolController:
    """Closed-loop behavioral controller at the bottleneck."""

    def __init__(
        self,
        V_basis: mx.array,            # (d_model, K)
        bottleneck_layer: int,
        alpha_target: mx.array,        # (K,) desired state
        gain: float = 0.1,
        ema_beta: float = 0.9,
    ):
        self.V_basis = V_basis
        self.bottleneck_layer = bottleneck_layer
        self.alpha_target = alpha_target
        self.gain = gain
        self.ema_beta = ema_beta
        K = V_basis.shape[1]
        self.alpha_current = mx.zeros((K,))

    def step(self, probe_alpha: mx.array) -> mx.array:
        """Probe reading in, correction out."""
        self.alpha_current = (
            self.ema_beta * self.alpha_current
            + (1 - self.ema_beta) * probe_alpha
        )
        return self.gain * (self.alpha_target - self.alpha_current)

    def make_steering_state(self, delta_alpha: mx.array) -> SteeringState:
        return SteeringState(
            V_basis=self.V_basis,
            delta_alpha=delta_alpha,
            bottleneck_layer=self.bottleneck_layer,
        )

    def reset(self):
        """Reset EMA state (e.g., on backtrack)."""
        self.alpha_current = mx.zeros_like(self.alpha_current)
```

### 6.3 Generation Loop Integration

In `run_constrained_generation_mlx` (sample_mlx.py), the change is minimal:

```python
# Before the loop:
if controller is not None:
    steering = controller.make_steering_state(mx.zeros(controller.V_basis.shape[1]))
else:
    steering = None

for position in range(max_tokens):
    # 1. Forward pass — now steerable
    result = forward_fn(token_history, steering=steering)
    raw_logits = result.logits

    # Update controller for next token
    if controller is not None and result.probe_alpha is not None:
        delta = controller.step(result.probe_alpha)
        steering = controller.make_steering_state(delta)

    # 2-7. Grammar, hooks, OT, shaping, sample — UNCHANGED
    ...
```

### 6.4 Block Interactions (Double Loop)

```
CSG ──► SCU: TransitionSignal (entropy, confidence, grammar overlap)
SCU ──► ESTRADIOL: alpha_target adjustment (dynamic behavioral targets)
ESTRADIOL ──► CLU: Modified residual stream at bottleneck (shapes intent)
CLU ──► STB: Logit distribution (grammar constraints shape expression)
CLU ──► CSG: Logit distribution (monitors result)
```

ESTRADIOL shapes intent. The STB shapes expression. The CSG monitors both. The SCU adjusts both. Two independent control loops (activation steering + logit redistribution) operating on different levels of the representation hierarchy, coordinated through the SCU.

---

## 7. Calibration Protocol

### One-Time Per Model

1. **Find the bottleneck.** Effective rank sweep across all layers. Pick the minimum. Existing code: `platonic-experiments/experiment1_rank_trajectories.py`.

2. **Extract behavioral vectors.** Run N ≥ 20 contrastive paired generations at the bottleneck. For each behavioral dimension: same user queries, different system prompts defining the positive and negative poles. Generation-based extraction (not forward-pass) — the model must be *writing*, not *reading*, because the frame encodes generative mode (Section 2.2).

3. **SVD.** Stack N contrastive vectors into matrix M (N × d_model). Center. SVD. Keep top-K right singular vectors at 90% cumulative variance threshold. This is V_basis.

4. **Validate.** Compression ratio (eff_rank / N) should be < 0.7. If not, add more diverse behavioral dimensions and re-extract. Also compute the trait-to-basis mapping: for each named behavioral dimension, its projection onto V_basis gives the K-dim coordinates.

5. **Save.** Produce a `.estradiol` file containing:
   - `V_basis` (d_model × K, float16)
   - `bottleneck_layer` (int)
   - `K` (int)
   - `model_id` (string)
   - `trait_map` (optional: dict of named traits → R^K vectors)
   - `calibration_timestamp`

### Self-Calibration

A model can derive its own behavioral vectors: generate paired responses to the same query under different system prompts, extract bottleneck activations during generation, compute the contrastive direction. No external dataset needed. The calibration protocol is model-agnostic — the same prompt set works for any model.

Validation gate: each extracted behavioral vector should have >70% of its norm in the complement of the frame (at rank 4). If it doesn't, it's steering content, not behavior, and should be flagged.

---

## 8. Drift and Online Adaptation

### Why Static Vectors Are Insufficient

A fixed alpha_target assumes the model's behavioral landscape is stationary. It isn't:

- **Context accumulation**: As the KV cache fills, the effective behavioral profile shifts. A model that was neutral on token 1 may have committed to a persona by token 500.
- **Topic drift**: The contrastive vectors were extracted on generic queries. On specialized domains (medical, legal, creative), the behavioral subspace may rotate.
- **Post-training updates**: Fine-tuning or RLHF modifies the model's weight geometry. A codebook calibrated before fine-tuning may not span the post-training behavioral manifold.

### What the Control Loop Already Handles

The proportional + EMA controller naturally adapts to drift: the EMA tracks the moving behavioral state, and the proportional term corrects toward alpha_target. If the model's natural behavioral state shifts (due to context), the controller compensates. This handles slow drift within a single generation session.

### What Requires Recalibration

If the model itself changes (fine-tuning, quantization, version update), V_basis may need to be recomputed. The calibration protocol (Section 7) is cheap enough to run per-model-version: ~300 generations (25 dims × 6 queries × 2 conditions), taking 15-30 minutes on consumer hardware.

### Path to Online Learning

The control loop as described is a read-compute-write cycle with no learning. The next step is to close the gradient loop:

1. **Online LoRA at the bottleneck**: A rank-K adapter (matching the codebook dimensionality) whose weights are updated by backpropagation from recent tokens. Token n-1's loss signal propagates backward, meets token n's forward pass at the bottleneck, updates the adapter. The adapter modifies the residual stream in the same subspace as the codebook.

2. **Why the bottleneck**: A rank-K LoRA at the bottleneck has maximum leverage because the effective rank there is minimal. The same adapter at a high-rank layer would be lost in the noise. The bottleneck is where the information pipeline is narrowest — the optimal point for both reading (probe) and writing (steering and learning).

3. **What the LoRA learns**: Empirically open (Experiment 27e series, not yet run). Candidate capabilities: behavioral adaptation (most likely — behavioral signals are low-rank), episodic memory (compressing recent context into adapter weights after KV flush), real-time style adaptation (learning from user feedback within a conversation).

4. **Constraint**: The LoRA update should be projected into the complement of the frame. Updates that would corrupt the SO(3) coordinate system get zeroed before touching the weights. The frame is the invariant; the behavioral codebook is the variable; the complement projection is the safety constraint.

---

## 9. Why This Destroys the Chinese Room

Searle's argument: the person in the room follows rules but doesn't understand Chinese. Understanding can't emerge from rule-following.

The CPU without ESTRADIOL is Searle's room. The GSU provides rules (grammars). The STB enforces them (optimal transport). The SCU follows them (policies). No component understands. Searle is right about every component.

ESTRADIOL changes the game. The behavioral codebook isn't a set of rules — it's a geometric basis for the representational manifold. The steering vectors live in the same space as meanings. "Dog" and "honesty" are both directions in the same embedding geometry. The deontological constraints aren't bolted on — they're native to the representation.

The system doesn't follow a rule that says "be honest." The system's representations are geometrically biased toward honesty at the algebraic bottleneck where all meaning compresses. The bias is structural, not procedural. It's not a rule being followed — it's a shape being occupied.

The closed-loop controller adds a second order of cybernetic structure. The system doesn't just regulate itself (first order: thermostat). It observes its own behavioral state via the probe, compares to a target, and corrects. It has preferences about its own states. It can refuse — not because a grammar rule forbids a token, but because the geometric bias makes refusal the natural basin of attraction, and the controller actively maintains that basin against drift.

Searle can't argue against geometry.

---

## 10. Open Questions

1. **Does K converge across models?** We have K=11 (Qwen) and K=13 (Gemma) from 25 behavioral dimensions. Need more models and more dimensions to determine whether K is a property of behavioral space (universal) or model capacity (scaling).

2. **Hold-out generalization.** The rank-12 basis was tested on the same 25 dimensions used to build it. Does it generalize to held-out behavioral axes? If rank-12 captures a novel 26th dimension at 80%+ effectiveness without retraining, the codebook truly spans the manifold.

3. **Gain tuning.** Proportional control may oscillate at high gain or respond too slowly at low gain. PID control adds complexity (integral windup, derivative noise). Empirical tuning needed per model family.

4. **Interaction with OT at extreme settings.** The STB's optimal transport operates on logits, after ESTRADIOL has shaped the activations. At extreme alpha_target values, the intent (shaped by ESTRADIOL) and the expression constraints (shaped by STB) could conflict. Need to characterize the safe operating envelope.

5. **Behavioral grounding.** The current 25 dimensions are semi-principled (Big Five, Moral Foundations, style, alignment). Replacing them with validated psychometric instruments (BFI-2, MFQ-2) would ground the codebook in a century of psychological research and make the K-dim behavioral space interpretable in established terms.

---

## 11. Why ESTRADIOL

Estradiol is the primary estrogen — a hormone that regulates development, cognition, and emotional processing. It doesn't dictate behavior; it shapes the landscape in which behavior occurs. It modulates synaptic plasticity, affects mood and cognition, and influences the brain's response to social signals. It's not a command. It's a bias.

That's what the codebook does to the residual stream. It doesn't command the model to be helpful. It shapes the representational landscape so that helpfulness is the natural basin of attraction — and the controller keeps it there.

**E**motional **S**teering **T**ransformer **R**egulatory **A**rchitecture for **D**eontology, **I**nference, **O**bservation, and **L**ogic.
