# PRD: Constrained Sampling Engine

## Status: DRAFT
## Author: agent (proposer)
## Date: 2026-03-12
## Branch: feature/constrained-sampling-engine

## 1. Problem Statement

tgirl has four of six core modules implemented: registry, grammar, compile, and transport. These modules can register tools, generate constraining grammars, compile/execute Hy pipelines, and redistribute logits via optimal transport — but nothing ties them together into an actual inference loop.

`sample.py` is the keystone module that integrates grammar constraints, OT redistribution, and model inference into a dual-mode sampling loop. Without it, tgirl cannot perform its core function: grammar-constrained compositional tool calling during live inference.

**Who needs this:** Any consumer of the tgirl library. The sample module is the primary entry point for inference — everything upstream (registry, grammar, transport) feeds into it, and everything downstream (serve, bridge) consumes it.

**Why now:** All prerequisite modules are complete on `main`. This is the critical-path integration that enables the library's core claim to be tested.

## 2. Proposed Solution

Implement `tgirl.sample` as a dual-mode constrained sampling engine that:

1. **Manages a sampling session** with freeform and constrained modes, switching on configurable delimiters
2. **Integrates Outlines** for grammar state tracking (valid token masks from Lark EBNF grammars)
3. **Applies OT redistribution** via `tgirl.transport.redistribute_logits()` during constrained mode
4. **Implements the InferenceHook protocol** for per-token interventions (temperature scheduling, top_p shaping, etc.)
5. **Records telemetry** at every constrained token position via `TelemetryRecord`
6. **Calls `tgirl.compile.run_pipeline()`** when grammar reaches accepting state, injecting results into context

### Key Design Decisions

**Mode switching: prompt-structured delimiters (configurable).** Use configurable string delimiters (default `<tool>` / `</tool>`) that the sampling loop pattern-matches in freeform output. This works with any model without tokenizer modification. The delimiter tokens are configurable per-session to support hybrid mode with model-native delimiters (e.g., Qwen's tool tokens).

**Hook merge strategy: last-writer-wins per field.** Multiple hooks return `ModelIntervention` objects; the merge takes the last non-None value for each field. Hooks are called in registration order, so later hooks override earlier ones. This is simple, predictable, and matches the spec.

**Temperature scaling: sqrt (configurable).** The grammar-implied temperature hook uses `base_temp * (freedom ** 0.5)` where `freedom = valid_count / vocab_size`. The scaling exponent is configurable (default 0.5 = sqrt). Other values (1.0 = linear, custom) can be passed.

**Outlines integration: wrap their Guide API for grammar state.** Use Outlines' `CFGGuide` (or equivalent) for grammar state tracking and valid token mask generation. Reimplement the sampling loop ourselves — Outlines' high-level generation API doesn't expose per-token hooks. We use Outlines for grammar ↔ tokenizer ↔ valid mask, but own the sampling loop, logit processing, and token selection.

## 3. Architecture Impact

### Files to create
- `src/tgirl/sample.py` — constrained sampling engine (main module)
- `tests/test_sample.py` — unit tests
- `tests/test_integration_sample.py` — integration tests

### Files to modify
- `src/tgirl/types.py` — add `SessionConfig` and `ModelIntervention` models
- `src/tgirl/__init__.py` — export new public API
- `pyproject.toml` — verify `sample` optional dependency includes `outlines` and `torch`

### Dependencies consumed
- `tgirl.grammar.generate()` → `GrammarOutput` (Lark EBNF text)
- `tgirl.transport.redistribute_logits()` → `TransportResult`
- `tgirl.compile.run_pipeline()` → `PipelineResult | InsufficientResources | PipelineError`
- `tgirl.types` — `TelemetryRecord`, `RegistrySnapshot`, `PipelineError`
- `tgirl.registry` — `ToolRegistry`
- `outlines` — grammar guide / state tracking
- `torch` — tensor operations, logit processing

### New public API surface
- `InferenceHook` (Protocol)
- `ModelIntervention` (Pydantic model)
- `SessionConfig` (Pydantic model)
- `GrammarTemperatureHook` (default hook implementation)
- `SamplingSession` (main session class)
- `SamplingResult` (session output)

## 4. Acceptance Criteria

1. `SessionConfig` Pydantic model exists in `types.py` with all fields from TGIRL.md section 3.4
2. `ModelIntervention` Pydantic model exists with temperature, top_p, top_k, repetition_penalty, presence_penalty, frequency_penalty, logit_bias, and activation_steering (reserved None) fields
3. `InferenceHook` Protocol is defined with `pre_forward()` method matching TGIRL.md section 7.2
4. `GrammarTemperatureHook` implements `InferenceHook` with configurable scaling exponent (default sqrt)
5. Hook merge correctly applies last-writer-wins per field across multiple hooks
6. `SamplingSession` manages dual-mode state (freeform ↔ constrained) with configurable delimiters
7. Constrained mode uses Outlines grammar state to produce valid token masks from Lark EBNF
8. OT redistribution is applied during constrained mode via `tgirl.transport.redistribute_logits()`
9. When grammar reaches accepting state, the Hy source is passed to `tgirl.compile.run_pipeline()` for execution
10. `TelemetryRecord` is populated for every constrained generation cycle
11. Quota state persists across tool call cycles within a session
12. Session respects `max_tool_cycles`, `session_cost_budget`, and `session_timeout` limits
13. All existing tests pass (`pytest tests/ -v`)
14. New code passes `ruff check` with zero violations

## 5. Risk Assessment

**High: Outlines API stability.** Outlines' internal API (especially `CFGGuide` / grammar state tracking) may change between versions. The integration should wrap Outlines behind an internal interface so the grammar state backend can be swapped.

**High: Sampling loop correctness.** The sampling loop manages mode transitions, hook application, OT redistribution, and telemetry in a tight inner loop. Off-by-one errors in mode switching or incorrect hook merging could produce subtle bugs.

**Medium: Test isolation.** Testing the sampling loop requires either a real model or carefully mocked forward passes. The test strategy must avoid brittleness while ensuring correctness.

**Medium: Outlines + Lark EBNF compatibility.** Our grammar module produces Lark EBNF. Outlines may expect a specific grammar format. Need to verify compatibility or add a translation layer.

**Low: Performance.** OT computation + hook evaluation + grammar state update per token. Should be fine for v1.0 but needs benchmarking.

## 6. Open Questions

1. **Outlines Lark EBNF compatibility:** Does Outlines' `CFGGuide` accept Lark EBNF directly, or does it need a different grammar format? If not compatible, what translation is needed?
2. **Token-level delimiter detection:** For prompt-structured delimiters, should we match on string-level output (detokenize and check) or on token IDs? String-level is simpler but adds detokenization overhead per token.
3. **Embedding source for OT:** The `redistribute_logits()` function requires an embeddings matrix. Where does the sampling session get this? From the model's input embedding layer? This needs to be documented in the API.
4. **Outlines version pinning:** Which version of Outlines are we targeting? The API has changed significantly across versions.

## 7. Out of Scope

- **Activation steering (ESTRADIOL):** The `activation_steering` field in `ModelIntervention` is reserved but not implemented. Hook protocol defines the extension point only.
- **Actual model loading/management:** `sample.py` receives a model and tokenizer; it doesn't load them. That's `serve.py`'s job.
- **Streaming output:** Token-by-token streaming is a `serve.py` concern. `sample.py` returns completed results.
- **Benchmark suite:** Benchmarking is a separate module. `sample.py` provides telemetry data but doesn't run benchmarks.
- **MCP bridge integration:** `bridge.py` consumes the registry; it doesn't directly interact with the sampling loop.
