# VCogPU Engineering Roadmap

**Version:** 0.3 — March 2026
**Architecture:** ARCHITECTURE.md (block definitions), ESTRADIOL.md (activation steering design)
**Current:** v0.1.0 — core inference engine operational, LinGO data pipeline complete

---

## Thesis

tgirl is a **grammar-aware inference engine** — it takes any polyglot language model and boosts performance through inference-time grammar constraints, auto-tuning hyperparameters, and optimal transport logit redistribution. For domains where output structure is formally specifiable (programming languages, tool calling, grammatical natural language, structured data), injecting structural knowledge at inference time is strictly better than scaling models. Grammar constraints eliminate structural errors by construction. OT preserves the model's semantic intent. The ADSR modulation matrix auto-tunes per-token parameters.

**Evidence:** Qwen3.5-9B (instruct, 4-bit quantized on Apple Silicon) achieves 80% AST accuracy on BFCL v4 with tgirl — matching Opus 4.5, +14pp above the model's reported 66%. The model has native tool-calling support from post-training (SFT + RL), but cannot reliably produce structurally correct output without tgirl's grammar constraints. The inference loop amplifies competence the model already possesses.

**Target:** Grammar-agnostic inference engine supporting CFGs for programming languages (Python, Rust, JS, etc.), HPSG for natural language, tool-calling grammars for agentic behavior, and schema grammars for structured output. Any grammar × any model × any hardware. This translates directly to codegen benchmarks (HumanEval, MBPP, SWE-bench), tool-calling benchmarks (BFCL), and structured output benchmarks.

---

## Block Status

| Block | Module(s) | Status | Key Gap |
|-------|-----------|--------|---------|
| CLU | `cache.py` | Complete | No intermediate layer hooks |
| RSF | `cache.py` | Complete | — |
| SKB | External (model weights) | Complete | — |
| GSU | `grammar.py`, `outlines_adapter.py`, `lingo/` | CFG complete, HPSG data pipeline complete | No active HPSG constraint masks, no learned controller |
| SCU | `state_machine.py` | Complete (5 policies) | No multi-grammar weight vectors |
| STB | `transport.py`, `transport_mlx.py` | Complete for single grammar | No weighted multi-grammar masks |
| CSG | `sample.py`, `sample_mlx.py`, `modulation.py` | Complete (6 signals incl. coherence) | No MLP-derived type distribution signal |
| IOC | `compile.py`, `bridge.py`, `serve.py`, `cli.py` | Complete | — |
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

## Phase 2: LinGO-Native GSU — Grammar-Aware Inference

**Priority:** High — the foundation for grammar-agnostic inference
**Status:** Phase 2A complete. Phases 2B–2D are the path to active grammar constraints.

The GSU extends from CFG-only to a polyglot grammar engine supporting HPSG (natural language), CFGs (programming languages, tool calling, structured output), and future formalisms. The progression is A → B → C, where each step makes the next easier.

### Phase 2A: Data Pipeline + Coherence Signal ✅ COMPLETE

TDL parser, type hierarchy (51k types, O(1) subsumption), lexicon (43k entries), token-to-lexeme mapping, coherence signal, GrammarState adapter, modulation matrix integration. All implemented in `tgirl.lingo/`. The coherence signal provides *soft* modulation — "how grammatical is the recent output" — but no hard masking.

**Module:** `src/tgirl/lingo/` — `tdl_parser.py`, `types.py`, `lexicon.py`, `grammar_state.py`
**Tests:** 80 lingo tests + modulation regression
**Branch:** `feature/lingo-native` (pending merge to main)

### Phase 2B: Learned Syntactic Constraint Predictor (MLP)

**Priority:** High — turns passive coherence into active prediction

The model's logit distribution already encodes grammatical knowledge implicitly. A small MLP extracts that knowledge and projects it into the ERG's type space.

```
logits (vocab_size) → softmax → TokenLexemeMap projection → type distribution (~956 leaf types)
    ↓
MLP (~956 → hidden → ~956)
    ↓
predicted valid lexeme types → TokenLexemeMap inverse → token mask
```

The MLP input is tiny because TokenLexemeMap compresses vocab → ~956 ERG leaf lexeme types. The MLP learns type-transition probabilities — a learned approximation of the ERG's construction rules.

**Training signal:**
- Construction rules in `constructions.tdl` (293 rules) define valid type sequences
- Treebanked corpora (Redwoods) provide empirical type-transition frequencies
- Self-supervision: run text through type hierarchy, collect (current_type_distribution → next_type) pairs

**Output:** The MLP feeds the modulation matrix (replacing/augmenting the coherence signal) or produces a real token mask via inverse TokenLexemeMap projection. First version is soft (signal → modulation), second is hard (mask → GSU).

### Phase 2C: HPSG → CFG Extraction

**Priority:** Medium — bridges HPSG to llguidance's CFG engine

Extract a context-free backbone from the ERG:
- Construction rules (293) → CFG productions
- Lexeme type leaves (~956) → terminals
- Feature structures dropped (overapproximation — the CFG permits more than the HPSG)
- Feed the resulting CFG to llguidance

This gives the GSU two grammar types: the Hy/tool-calling CFG and the NL-derived CFG. Both produce masks through the same `GrammarState` interface. The MLP controls which is active via the grammar weight vector.

### Phase 2D: Incremental Chart Parsing

**Priority:** Low (research/v2) — precise but expensive

Full incremental HPSG parser. Advance a chart after each token. Valid next tokens = tokens that extend at least one active edge. The MLP (from 2B) prunes the chart by predicting which edges the model is actually pursuing — keeps parsing tractable.

Target: <1ms/token for pruned chart advancement on ERG.

---

## Phase 3: Multi-Grammar GSU + Codegen Grammars

**Priority:** High — the codegen benchmark path
**Depends on:** Phase 2B (needs learned controller for grammar weights)

### Grammar Portfolio

The GSU becomes a portfolio of grammars that can be activated simultaneously with learned weights:

| Grammar | Source | Use Case |
|---------|--------|----------|
| Hy CFG | `tgirl.grammar` (Jinja2 templates) | Tool calling, composition |
| English NL | ERG 2025 (Phase 2B/C) | Natural language well-formedness |
| Python CFG | Python grammar spec | Python codegen |
| Rust CFG | Rust grammar spec | Rust codegen |
| JSON Schema | JSON Schema → CFG | Structured output |
| TypeScript CFG | TS grammar spec | Web codegen |

Each grammar compiles to the same `GrammarState` interface. Adding a new grammar requires no changes to the inference loop — only a new grammar state provider.

### MLP-Controlled Grammar Selection

The MLP from Phase 2B expands to control the `GrammarWeightVector`:
- Input: logit distribution projected through grammar-specific token maps
- Output: per-grammar activation weights
- Training: self-supervised from generation quality metrics per grammar

During freeform generation: NL grammar active, tool grammar weight → 0.
During constrained generation: tool grammar active, NL grammar provides soft guidance.
During codegen: language-specific CFG active, NL grammar off.

### SCU Extension

- `GrammarWeightVector` — N weights, one per active grammar
- `GrammarWeightPolicy` — MLP-driven weight updates per token
- SCU manages active grammar set, provides weighted masks to STB

### STB Extension

- `redistribute_logits` accepts per-grammar masks with weights
- Combined valid mask: weighted sum of per-grammar boolean masks
- OT concentrates the model on tokens satisfying the active grammars
- Cost matrix unchanged — embedding distance is grammar-agnostic

### Codegen Benchmark Targets

| Benchmark | Grammar | Target |
|-----------|---------|--------|
| BFCL v4 | Hy CFG | >85% AST accuracy (9B base) |
| HumanEval | Python CFG | Measure structural error elimination |
| MBPP | Python CFG | Measure on constrained small models |
| MultiPL-E | Multi-language CFGs | Cross-language grammar switching |

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
Phase 1 (IOC) ─────────── COMPLETE
Phase 2A (LinGO data) ──── COMPLETE
Phase 2B (MLP controller)─ can start now
Phase 2C (HPSG→CFG) ────── depends on 2B
Phase 2D (Chart parsing) ── depends on 2C (research)
Phase 3 (Multi-Grammar) ── depends on 2B (needs learned controller)
Phase 4 (ESTRADIOL) ────── depends on Phase 1 (CLU hooks)
Phase 5 (IRQ v2) ───────── depends on Phase 1 (IOC foundation)
Phase 6 (Research) ──────── depends on Phase 2C + Phase 3
```

---

## Benchmarks

Results tracked per phase. The goal is grammar-agnostic performance gains across all structured output domains.

### Current Results (v0.1.0 + ADSR)

| Benchmark | Model | Metric | Value |
|-----------|-------|--------|-------|
| BFCL v4 simple_python | Qwen3.5-9B (instruct, 4-bit MLX) | AST accuracy | **80%** (314/391) — matches Opus 4.5 |
| BFCL v4 simple_python | Qwen3.5-9B | vs. reported baseline | +14pp (66% → 80%) |
| BFCL v4 simple_python | Qwen3.5-0.8B (instruct, 4-bit MLX) | AST accuracy | 48% (182/382) |
| BFCL v4 simple_python | Qwen3.5-0.8B (ADSR tuned) | AST accuracy (50-entry subset) | ~80% |
| Showcase (8 tools, 15 requests) | Qwen3.5-0.8B (instruct, 4-bit MLX) | Routing accuracy | 13/15 |

### Target Benchmarks (Phase 3+)

| Benchmark | Grammar Type | What We Measure |
|-----------|-------------|-----------------|
| BFCL v4 | Hy CFG | Tool calling accuracy (current: 80%) |
| HumanEval | Python CFG | Structural error elimination rate |
| MBPP | Python CFG | Small-model performance with constraints |
| MultiPL-E | Multi-language CFGs | Cross-language grammar switching |
| SWE-bench Lite | Python CFG + tool calling | End-to-end agent codegen |
