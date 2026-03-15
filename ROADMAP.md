# VCogPU Engineering Roadmap

**Version:** 0.2 — March 2026
**Architecture:** ARCHITECTURE.md (block definitions), ESTRADIOL.md (activation steering design)
**Current:** v0.1.0 — core inference engine operational

---

## Block Status

| Block | Module(s) | Status | Key Gap |
|-------|-----------|--------|---------|
| CLU | `cache.py` | Complete | No intermediate layer hooks |
| RSF | `cache.py` | Complete | — |
| SKB | External (model weights) | Complete | — |
| GSU | `grammar.py`, `outlines_adapter.py` | Complete for CFG | Single formalism only |
| SCU | `state_machine.py` | Complete (5 policies) | No multi-grammar weight vectors |
| STB | `transport.py`, `transport_mlx.py` | Complete for single grammar | No weighted multi-grammar masks |
| CSG | `sample.py`, `sample_mlx.py` | Complete (5 signals) | No `linguistic_coherence` signal |
| IOC | `compile.py`, `sample.py` | Partial | No `serve.py`, no `bridge.py` |
| IRQ | `sample.py` | Minimal | Timeout + injection only |
| ESTRADIOL | `types.py` (reserved field) | Not started | Needs CLU hooks first |

---

## Phase 1: IOC Completion — MCP Bridge + Local Server

**Priority:** High — enables external integration and testing

The IOC is the interface between the CPU and the external world. Without it, tgirl can only be exercised programmatically.

### bridge.py — MCP Compatibility Layer

- Import MCP tool schemas as tgirl tool registrations
- Export tgirl tool pipelines as MCP tool calls
- Enables tgirl to participate in MCP ecosystems
- Dependency already declared: `bridge = ["mcp>=1.0"]`

### serve.py — FastAPI Local Inference Server

- REST/WebSocket API for dual-mode inference
- Wraps `SamplingSession` with HTTP interface
- Streaming token output via WebSocket
- Dependency already declared: `serve = ["fastapi>=0.110", "uvicorn>=0.28", "transformers>=5.0"]`

---

## Phase 2: LinGO Integration into GSU

**Priority:** High — extends grammar constraint surfaces to formal linguistics

The GSU currently supports only context-free grammars via Lark EBNF. Integrating HPSG grammars from the LinGO Grammar Matrix extends the architecture to formally verifiable linguistic competence — the grammar becomes a mathematically grounded constraint surface for natural language well-formedness, not just tool-calling syntax.

### Investigation (before implementation)

- Can PyDelphin perform incremental parsing? (per-token well-formedness checking required)
- If not: approximate linguistic coherence via sliding-window batch parsing or n-gram model over valid HPSG parses
- Performance target: <1ms/token for coherence evaluation

### Implementation

1. Add `pydelphin` as optional dependency: `hpsg = ["pydelphin>=1.0"]`
2. Create `hpsg_adapter.py` implementing the `GrammarState` protocol (`get_valid_mask`, `is_accepting`, `advance`)
3. Add `linguistic_coherence: float = 0.0` to `TransitionSignal`
4. Extend `compute_transition_signal_{torch,mlx}` to compute coherence from HPSG state
5. Update `ConfidenceTransitionPolicy` to weight the coherence signal

**Architecture constraint:** HPSG must produce the same `GrammarState` interface as CFG. The constrained generation loop doesn't change; only the grammar state provider does. The GSU is formalism-agnostic by design.

---

## Phase 3: Multi-Grammar SCU + Cross-Lingual STB

**Priority:** Medium — enables grammar composition and cross-lingual transport
**Depends on:** Phase 2 (needs a second grammar formalism)

### SCU Extension

- New type: `GrammarWeightVector` — N weights, one per active grammar
- New policy class: `GrammarWeightPolicy` — adjusts weights based on CSG signals
- SCU manages active grammar set and provides weighted masks to STB

### STB Extension

- `redistribute_logits` accepts per-grammar masks with weights
- Combined valid mask: weighted sum of per-grammar boolean masks
- Cost matrix unchanged — embedding distance is grammar-agnostic

### Validation Experiment

One model, two grammars (e.g., English CFG + Chinese CFG), a generation task requiring cross-lingual reasoning. Measure semantic preservation (embedding similarity) and language purity (target-language token ratio) with and without cross-lingual OT.

---

## Phase 4: ESTRADIOL — Activation Steering

**Priority:** Medium — second-order cybernetic upgrade
**Depends on:** Phase 1 (CLU hook interface)
**Design:** ESTRADIOL.md

### Phase 4a: CLU Hook Interface

- Extend `make_mlx_forward_fn` and `make_hf_forward_fn` to accept optional layer hooks
- Hook interface: `Callable[[int, Tensor], Tensor]` — receives (layer_index, activation), returns modified activation
- Intervention point: after bottleneck layer MLP output, before next layer's attention input

### Phase 4b: Contrastive Vector Extraction

- Tooling for paired-prompt activation extraction at the algebraic bottleneck
- Steering vectors for core dimensions: honesty, helpfulness, safety
- Calibrate bottleneck layer for each target model (methodology: effective rank trajectory from Platonic Experiments)

### Phase 4c: SCU Integration

- `SteeringIntervention` type fills `ModelIntervention.activation_steering`
- `SteeringPolicy` in SCU controls α coefficient per steering vector
- CSG gains `semantic_risk` signal dimension for dynamic steering adjustment
- Double feedback loop: ESTRADIOL shapes intent (pre-logit), STB shapes expression (post-logit), CSG monitors both

---

## Phase 5: Interrupt-Driven IOC

**Priority:** Low — paradigm shift, not incremental improvement
**Depends on:** Phase 1 (IOC foundation)

Replace the chat-template request-response model with a continuous inference daemon.

- Model runs continuously; `stream_to_user` is an I/O tool
- User input is an interrupt, not a prompt
- Tool calls are I/O operations dispatched through the IOC
- The IRQ block becomes the primary external interface
- Separation: external events (IRQ) vs internal state transitions (SCU)

This is a v2.0 architectural change.

---

## Phase 6: Emergent Contact Grammars (Research)

**Priority:** Research — exploration, not engineering
**Depends on:** Phase 2 + Phase 3

When multiple linguistic grammars are simultaneously active with non-trivial weights, characterize the contact grammars that emerge from the valid token intersection. This is a research investigation enabled by the architecture, not a planned engineering deliverable.

---

## Dependencies

```
Phase 1 (IOC) ─────────── can start now
Phase 2 (LinGO) ────────── can start now (parallel with Phase 1)
Phase 3 (Multi-Grammar) ── depends on Phase 2
Phase 4 (ESTRADIOL) ────── depends on Phase 1 (CLU hooks)
Phase 5 (IRQ v2) ───────── depends on Phase 1 (IOC foundation)
Phase 6 (Research) ──────── depends on Phase 2 + Phase 3
```

---

## Benchmarks

Results will be tracked per phase. Current baseline (v0.1.0):

| Benchmark | Model | Metric | Value |
|-----------|-------|--------|-------|
| BFCL v4 simple_python | Qwen3.5-0.8B (base, no instruct) | AST accuracy | 48% (182/382) |
| BFCL v4 simple_python | Qwen3.5-0.8B | Tool call rate | 96% (382/400) |
| BFCL v4 simple_python | Qwen3.5-0.8B | Errors | 17/400 |
| BFCL v4 simple_python | Qwen3.5-0.8B | Avg latency | 5.4s/entry |
| Showcase (8 tools, 15 requests) | Qwen3.5-0.8B | Routing accuracy | 13/15 |
