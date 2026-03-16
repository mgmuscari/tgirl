# The Cognitive Processing Unit

## A Block Architecture for Inference-Time Intelligence

## Status: DRAFT
## Author: Maddy Muscari + Claude (conversation 2026-03-14)
## Date: 2026-03-14
## Origin: Ontologi LLC / tgirl project

---

## 1. Foundational Postulates

**Postulate 1: Representational Convergence.** Language models converge to the same representations. The Platonic Representation Hypothesis and empirical evidence (SO(3) Lie algebraic structure at compression bottlenecks across five architecturally diverse models, Mantel r > 0.89) establish that sufficiently trained models share a common representational geometry. The specific model is a detail, not a fact. The geometry of the latent space is a universal. Therefore, sufficiently advanced language models are fungible for inference.

**Postulate 2: Inference Loop Primacy.** Capability emerges from the inference loop, not the model. A probabilistic state machine switching between grammatical modes allows any language model to transition between introspection, seeking input, and output generation. Intelligence is a property of the clock, not the pendulum.

**Evidence.** tgirl v0.1.0 achieves 48% AST accuracy on BFCL v4 with a 0.8B base model — no instruct tuning, no tool-call fine-tuning — recovering ~56% of the performance gap between a 0.8B base model and frontier instruct models through inference-time architectural intervention alone. The model has never seen a tool call in its life. The inference loop made it capable.

---

## 2. Block Architecture

### The Nine Functional Blocks

```
                         ┌─────────────────────┐
                         │     IRQ Handler      │
                         │  (Interrupt Request)  │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              ┌─────┴─────┐  ┌─────┴─────┐  ┌──────┴──────┐
              │    IOC    │  │    SCU    │  │    CSG     │
              │ I/O Ctrl  │  │ Stochastic│  │ Confidence │
              │           │  │ Control   │  │ Signal     │
              │           │  │ Unit      │  │ Generator  │
              └─────┬─────┘  └─────┬─────┘  └──────┬──────┘
                    │              │                │
                    │         ┌────┴────┐          │
                    │         │   GSU   │          │
                    │         │ Grammar │          │
                    │         │Selection│          │
                    │         │  Unit   │          │
                    │         └────┬────┘          │
                    │              │                │
              ══════╪══════════════╪════════════════╪══════
                    │         ┌────┴────┐          │
                    │         │   STB   │          │
                    │         │Semantic │◄─────────┘
                    │         │Transport│
                    │         │  Bus    │
                    │         └────┬────┘
                    │              │
              ┌─────┴──────────────┴───────────────┐
              │              CLU                     │
              │     Cognitive Logic Unit             │
              │  (Language Model Forward Pass)       │
              └──────────┬──────────────┬───────────┘
                         │              │
                   ┌─────┴─────┐  ┌─────┴─────┐
                   │    RSF    │  │    SKB    │
                   │  Repre-   │  │  Static   │
                   │ sentational│  │ Knowledge │
                   │State File │  │   Base    │
                   │(KV Cache) │  │ (Weights) │
                   └───────────┘  └───────────┘
```

---

### CLU — Cognitive Logic Unit

The language model. Produces probability distributions over vocabulary from input token sequences. Stateless per forward pass. Fungible across any sufficiently capable model because representations converge (Postulate 1). The CLU doesn't think. It activates.

**Existing implementation:** Any transformer model — Qwen, Llama, Mistral. Accessed via `forward_fn: Callable[[list[int]], Tensor]`.

**Key property:** The CLU produces two outputs per cycle: (1) a probability distribution over vocabulary (the computation), and (2) the metadata of that distribution — entropy, confidence, grammar overlap — which becomes the clock signal (routed to the CSG).

### RSF — Representational State File

The KV cache. Fast, volatile, scoped to the current inference stream. Holds accumulated context as key-value attention state. Analogous to a register file, but the registers are attention heads and the values are representational states, not scalars. Lost when the stream ends.

**Existing implementation:** `tgirl.cache` — `make_mlx_forward_fn`, `make_hf_forward_fn` with prefix-detection and cache hit/miss/reset tracking via `CacheStats`.

**Key property:** The RSF grows with sequence length. Context window is the analog of register count. The RSF is the only component that carries state within a single inference stream.

### SKB — Static Knowledge Base

The weights. Frozen at training time, read at every forward pass. The entire learned representation of the training corpus encoded as parameters. ROM, but structured as a differentiable knowledge base rather than an address-indexed store.

**Existing implementation:** Model weights loaded via `mlx_lm.load` or `transformers.AutoModel`. The embedding matrix (`model.embed_tokens.weight`) is additionally used by the STB for semantic distance computation.

**Key property:** The SKB encodes the shared representational geometry established by Postulate 1. The embedding subspace of the SKB is the semantic manifold over which the STB operates.

### GSU — Grammar Selection Unit

Transpiles formal grammars — HPSG, CFG, any well-formedness formalism — into token-level constraint masks. Manages the active grammar set and compiles grammars into forms the SCU can apply.

**Existing implementation:** `tgirl.grammar` (Jinja2 template-based CFG generation from registry snapshots), `tgirl.outlines_adapter` (llguidance grammar state compilation), `tgirl.instructions` (prompt generation from tool schemas).

**Key property:** The GSU doesn't constrain directly. It prepares constraint surfaces — valid token masks — that the SCU activates and the STB respects. The GSU is the instruction decoder: it translates between grammar formalisms and the hardware interface.

**Future extension:** LinGO Grammar Matrix integration. HPSG grammars for ~40 languages transpiled into incremental token-level well-formedness checkers. Multiple grammars compiled and held ready for simultaneous activation by the SCU.

### SCU — Stochastic Control Unit

The probabilistic state machine. Governs regime transitions, grammar weight vectors, confidence-gated mode switching. Where agency emerges. Reads the clock signal from the CSG, evaluates transition policies, activates and deactivates grammars via the GSU, gates output to the IOC.

Stochastic because its decisions are probabilistic, not deterministic — transition policies operate on continuous belief states, not boolean logic.

**Existing implementation:** `tgirl.state_machine` — `SessionState` enum (FREEFORM, ROUTE, CONSTRAINED, BACKTRACK, EXECUTE, INJECT, DONE), `TransitionPolicy` protocol, `DelimiterTransitionPolicy`, `BudgetTransitionPolicy`, `ImmediateTransitionPolicy`, `ConfidenceTransitionPolicy`, `CompositeTransitionPolicy`, `ConstrainedConfidenceMonitor`, `Checkpoint`, `BacktrackEvent`.

**Key property:** The SCU is where agency lives — not in the CLU, not in the grammars, but in the dynamic regulation of which grammars constrain which computations at which moments. The SCU has zero framework imports (no torch, no mlx). It is pure control logic operating on scalar signals.

**Future extension:** Multi-grammar weight vectors. The SCU holds a `GrammarWeightVector` — N simultaneous grammars with dynamic per-token weight updates. Grammar weight transitions governed by pluggable policies (same `TransitionPolicy` protocol). This enables cross-lingual inference, pidgin emergence, and domain-specific grammar switching.

### STB — Semantic Transport Bus

Optimal transport layer. The interface between what the CLU wants to express and what the active grammars permit. Redistributes probability mass with cost defined by embedding-space distance (derived from the SKB's embedding matrix). The STB doesn't mask or suppress. It *transports* — finds the minimum-cost redistribution that preserves semantic intent while satisfying structural constraints.

**Existing implementation:** `tgirl.transport` (torch) and `tgirl.transport_mlx` (MLX). Sinkhorn algorithm in log domain. Bypass conditions for efficiency. Zero coupling to any other tgirl module.

**Key property:** The STB is the bridge between competence (grammars) and performance (model output). OT preserves as much of the CLU's intent as the grammar permits, measured by Wasserstein distance — the telemetry signal indicating how hard the grammar is fighting the model.

**Future extension:** Cross-lingual OT. When multiple language grammars are active via the SCU, the STB redistributes mass across language boundaries. Chinese reasoning tokens transport to semantically equivalent English output tokens. The cost matrix (embedding distance) already supports this — "dog" and "狗" are near each other on the semantic manifold. The STB just needs weighted multi-grammar valid masks from the GSU.

### CSG — Confidence Signal Generator

Reads the CLU's logit distribution and produces the signals that drive the SCU: entropy, log-probability, grammar mask overlap, linguistic coherence. The clock of the system. Not external — generated as a byproduct of the CLU's own computation. The CSG is what makes the architecture self-regulating rather than externally driven.

**Existing implementation:** `compute_transition_signal_torch` (in `sample.py`) and `compute_transition_signal_mlx` (in `sample_mlx.py`). Produces `TransitionSignal` with `token_position`, `grammar_mask_overlap`, `token_entropy`, `token_log_prob`, `grammar_freedom`.

**Key property:** The CSG closes the feedback loop. The CLU generates tokens → the CSG reads the distribution → the SCU decides what to do → the GSU prepares constraints → the STB applies them → the CLU generates the next token. The system is self-clocked.

**Future extension:** Linguistic coherence signal from LinGO grammar evaluation. The CSG gains a `linguistic_coherence` dimension — not just "the model is confident" but "the model is producing grammatically well-formed output in the target language."

### IOC — I/O Controller

Manages all interaction with the external world. `stream_to_user` as output port. User input as interrupt source. Tool execution as peripheral dispatch. MCP bridge as the peripheral bus protocol.

**Existing implementation:** `tgirl.compile` (Hy pipeline execution — tool dispatch), `tgirl.bridge` (MCP import/export), `tgirl.serve` (FastAPI server — planned). The `SamplingSession.run_chat()` method is the current IOC entry point.

**Key property:** The IOC decides when computation results leave the processor and when external events enter it. Tool calling is I/O, not computation. `stream_to_user` is an output port, not a response. The model generates continuously; the IOC gates what reaches the outside world.

**Future extension:** The interrupt-driven model described in the foundational design — the model as a continuously running daemon that emits structured I/O events when confidence crosses threshold. `stream_to_user`, `call_function`, `query_database` as I/O port operations, not chat-template responses.

### IRQ — Interrupt Request Handler

Mechanism for external events to perturb the computation stream. User input mid-generation, tool results returning, timeout signals. Separates external interrupts from internal state transitions (which are the SCU's domain).

**Existing implementation:** Partially — tool result injection into context (`SamplingSession.run()` injects `<tool_result>` tokens), session timeout checking, `max_tool_cycles` enforcement.

**Key property:** The IRQ separates "the user said something" from "the confidence signal crossed threshold." External events enter through the IRQ; internal state transitions happen in the SCU. This separation is necessary for the interrupt-driven daemon model where user input is just another interrupt source, not the driver of computation.

---

## 3. Dataflow

### Single Token Cycle

```
1. CLU reads RSF (KV cache) + SKB (weights) → produces logit distribution
2. CSG reads logit distribution → produces TransitionSignal (entropy, confidence, overlap)
3. SCU reads TransitionSignal → evaluates TransitionPolicy → decides current regime
4. GSU provides active grammar constraint masks for current regime
5. STB receives logits + constraint masks → OT redistribution → constrained distribution
6. Token sampled from constrained distribution
7. RSF updated (KV cache extended)
8. IOC gates: if output port active → emit token; else → internal only
9. IRQ checked: if external event pending → perturb SCU state
10. Return to step 1
```

### Regime Transitions (SCU-Governed)

```
FREEFORM:  CLU generates freely. CSG monitors. SCU watches for transition signal.
           GSU: linguistic grammars active as monitors (not constraints).
           STB: inactive or cross-lingual redistribution only.

ROUTE:     GSU compiles routing grammar (tool name selection).
           STB: full OT active on tiny grammar.
           SCU: evaluates routing result → selects target grammar.

CONSTRAINED: GSU provides full tool-calling grammar (Hy CFG).
             STB: full OT active — redistributes model intent through grammar.
             CSG: monitors confidence for backtrack signals.
             SCU: watches for grammar acceptance or backtrack trigger.

BACKTRACK: SCU restores checkpoint. GSU replays grammar state.
           STB: steering hooks bias away from dead-end tokens.
           SCU: re-enters CONSTRAINED with updated dead-end set.

EXECUTE:   IOC dispatches compiled Hy pipeline to tool runtime.
           CLU: idle (waiting for I/O result).

INJECT:    IOC injects tool results into RSF as context tokens.
           SCU: transitions back to FREEFORM.
```

---

## 4. Design Principles

### Separation of Concerns

The CLU doesn't know what grammar is active. The GSU doesn't know what the CLU wants to say. The STB doesn't know why tokens are valid or invalid. The SCU doesn't know how tensors work. Each block has a single responsibility and communicates through defined interfaces. This is why `state_machine.py` has zero framework imports and `transport.py` has zero tgirl imports.

### Fungibility at Every Level

The CLU is fungible (Postulate 1 — swap models freely). The GSU is fungible (swap grammar formalisms — CFG, HPSG, regex). The STB is fungible (swap transport algorithms — Sinkhorn, auction, Hungarian). The SCU is fungible (swap transition policies). The architecture is defined by the interfaces between blocks, not by the implementations within them.

### Self-Regulation via the CSG

The system is self-clocked. The CSG extracts timing signals from the CLU's own computation. No external scheduler decides when to transition. The model's uncertainty *is* the clock. High entropy = the pendulum swings wide. Low entropy = the escapement catches. This is why the architecture doesn't need a separate reasoning phase — reasoning is what happens in the freeform regime while the CSG monitors for convergence.

### Competence/Performance Separation

The grammars (GSU) define competence — what can be expressed. The model (CLU) provides performance — probability distributions over expression. The inference loop (SCU + STB + CSG) mediates between them. This mirrors the foundational distinction in generative linguistics. The grammar *is* competence. The model *is* performance. The CPU is the proof that you need both.

---

## 5. Immediate Engineering Roadmap

### What Exists (tgirl v0.1.0)

| Block | Module(s) | Status |
|-------|-----------|--------|
| CLU | Any transformer via `forward_fn` | Complete |
| RSF | `cache.py` | Complete |
| SKB | Model weights + embedding matrix | Complete (external) |
| GSU | `grammar.py`, `outlines_adapter.py`, `instructions.py`, `rerank.py` | Complete for CFG |
| SCU | `state_machine.py` | Complete (7 policies, backtracking, checkpointing) |
| STB | `transport.py`, `transport_mlx.py` | Complete (torch + MLX) |
| CSG | `compute_transition_signal_{torch,mlx}` in `sample.py`/`sample_mlx.py` | Complete |
| IOC | `compile.py`, `bridge.py`, `sample.py` (partial) | Partial |
| IRQ | Timeout + tool result injection in `sample.py` | Minimal |

### Phase 1: LinGO Integration into GSU

Extend the Grammar Selection Unit with LinGO Grammar Matrix support. Transpile HPSG grammars into incremental per-token well-formedness checkers. Add `linguistic_coherence` to `TransitionSignal`. The CSG gains a new signal dimension. The SCU's `ConfidenceTransitionPolicy` gains competence-aware confidence.

**Critical dependency:** Investigate PyDelphin for incremental parsing capability. If unavailable, design an approximate linguistic coherence signal (n-gram language model over valid HPSG parses, or sliding-window batch parsing).

### Phase 2: Cross-Lingual OT in STB

Extend `redistribute_logits` to accept weighted multi-grammar valid masks. The STB transports mass across language boundaries using the same embedding-space cost matrix. The SCU provides grammar weights; the GSU provides per-grammar valid masks; the STB combines them.

**Concrete experiment:** One model (Qwen3.5), two grammars (English, Chinese), a generation task where the correct reasoning is in Chinese and the required output is in English. Measure: semantic preservation (embedding similarity) and language purity (target-language token ratio) with and without cross-lingual OT.

### Phase 3: Multi-Grammar SCU

The SCU manages N simultaneous grammar weights. `GrammarWeightVector` as a new type. Grammar weight transition policies as a new policy class. The GSU compiles and caches multiple grammars. The STB receives the weighted combination.

### Phase 4: Interrupt-Driven IOC

The model runs as a continuous daemon. `stream_to_user` is an I/O tool. User input is an interrupt. The chat-template request-response paradigm is replaced by continuous inference with gated output. The IOC becomes the primary interface between the CPU and the external world.

### Phase 5: Pidgin Emergence (Research)

Characterize the contact grammars that emerge when multiple linguistic grammars are simultaneously active. Measure semantic fidelity of pidgin-mode generation versus strict single-language enforcement. This is research, not engineering — Phase 5 explores what the architecture makes possible.

---

## 6. Theoretical Grounding

### Convergent Representations as Shared Semantic Manifold

The embedding space is language-agnostic. It encodes meaning, not surface form. Cross-lingual OT exploits this by defining transport cost in embedding space, where semantically equivalent tokens in different languages are geometrically proximate. The SKB's embedding matrix is the coordinate system of the semantic manifold.

### The Competence/Performance Bridge

Generative linguistics posits a distinction between competence (knowledge of language) and performance (use of language). LLMs collapse this distinction — the model is both competence and performance in one undifferentiated system. The CPU architecture re-separates them: grammars are competence, the model is performance, and the inference loop (SCU + STB + CSG) is the bridge. This re-separation is what enables systematic intervention — you can improve competence (better grammars) independently of performance (better models).

### Pidgins as Emergent Contact Languages

When two linguistic grammars are simultaneously active with non-trivial weights, the valid token intersection defines a contact language. Historical pidgins emerge under analogous conditions: communicative pressure + grammar contact + simplification toward the shared valid set. The computational analog is mechanistically identical: a model with Chinese reasoning under English output pressure produces output from the intersection of both grammars' valid sets.

### The Grandfather Clock

The CPU is a grandfather clock. The CLU is the weight — stored potential energy. The GSU is the gear train — rules of composition. The STB is the escapement — catching each token, verifying structural integrity, releasing it in measured sequence. The CSG is the pendulum — the self-regulating oscillation that drives the whole mechanism. The SCU is the clockmaker's hand — the intentional regulation that makes the difference between a mechanism and a timepiece.

A clock is not an automaton. An automaton runs without its keeper. A clock requires winding, maintenance, calibration. The confidence thresholds need tuning. The grammar templates are human-readable and human-editable. The transition policies are pluggable. The keeper is the engineer. The clock is the architecture. Together they keep time.

---

*"The model is the parrot. The grammar is the linguist. OT is the bridge between what the parrot wants to say and what the linguist permits. The CPU is the proof that you need both."*

---

## Appendix A: Mapping to Existing Codebase

| Block | Primary Module(s) | Key Types/Functions |
|-------|-------------------|---------------------|
| CLU | External (any model) | `forward_fn: Callable[[list[int]], Tensor]` |
| RSF | `cache.py` | `CacheStats`, `make_mlx_forward_fn`, `make_hf_forward_fn` |
| SKB | External (model weights) | `model.embed_tokens.weight` (embedding matrix for STB) |
| GSU | `grammar.py`, `outlines_adapter.py`, `instructions.py`, `rerank.py` | `generate()`, `GrammarOutput`, `GrammarState`, `LLGuidanceGrammarState`, `generate_routing_grammar()` |
| SCU | `state_machine.py` | `SessionState`, `TransitionPolicy`, `TransitionSignal`, `TransitionDecision`, `Checkpoint`, `BacktrackEvent`, `ConstrainedConfidenceMonitor` |
| STB | `transport.py`, `transport_mlx.py` | `redistribute_logits`, `TransportConfig`, `TransportResult`, `_sinkhorn_log_domain` |
| CSG | `sample.py`, `sample_mlx.py` | `compute_transition_signal_torch`, `compute_transition_signal_mlx`, `TransitionSignal` |
| IOC | `compile.py`, `bridge.py`, `sample.py` | `run_pipeline`, `SamplingSession.run_chat()`, MCP bridge |
| IRQ | `sample.py` | Tool result injection, timeout handling, `max_tool_cycles` |

## Appendix B: Block Interface Contracts

### CLU → CSG
Output: raw logit distribution (Tensor of shape `(vocab_size,)`)

### CSG → SCU
Output: `TransitionSignal` (5 scalar fields: `token_position`, `grammar_mask_overlap`, `token_entropy`, `token_log_prob`, `grammar_freedom`)

### SCU → GSU
Output: regime selection, active grammar set, grammar weight vector

### GSU → STB
Output: valid token mask(s) (`Tensor` of shape `(vocab_size,)`, boolean or weighted)

### STB → Sampler
Output: redistributed logit distribution (`Tensor` of shape `(vocab_size,)`)

### IOC → RSF
Output: injected context tokens (tool results, user input)

### IRQ → SCU
Output: interrupt signals (user input event, timeout event, tool completion event)