# PRD: Grammar-Constrained Tool Re-Ranking

## Status: IMPLEMENTED
## Author: agent (proposer)
## Date: 2026-03-13
## Branch: feature/grammar-constrained-reranking

## 1. Problem Statement

Grammar-constrained generation guarantees syntactically valid tool calls, but it cannot guide **semantic** tool selection. At the token position where the tool name is chosen (position 2 in the s-expression), all registered tool names are equally valid grammar alternatives. The model must rely solely on its logit distribution to pick the right tool — and small models (0.8B-3B) frequently pick the wrong one.

Experimental results on Qwen3.5-0.8B with 7 JSON/string/math tools demonstrate:
- **Baseline (all tools in grammar):** 5/7 correct tool selection
- **Semi-structured re-ranking:** 6/7 (freeform mode picks tool, but fragile — model can go off-script)
- **Grammar-constrained re-ranking:** 7/7 (tiny grammar forces valid tool name output)
- **Log-prob scoring:** 1/7 (model priors dominate — not viable)

The grammar-constrained re-ranking approach achieves perfect tool selection even on a 0.8B model, with sub-second latency per re-rank (93-1333ms). It also yields a 3-8x speedup on the subsequent constrained generation pass because restricted grammars have far fewer valid tokens at each position.

This feature turns the experimental re-ranking from `examples/test_rerank.py` into a first-class, composable component of tgirl's sampling pipeline.

## 2. Proposed Solution

Add a **tool re-ranking pass** as an optional stage in the `SamplingSession` dual-mode flow:

1. When constrained mode is triggered (tool delimiter detected), generate a **routing grammar** — a minimal grammar that only accepts registered tool names.
2. Run a short constrained generation pass with the routing grammar and a **routing prompt** (derived from registry metadata) to select the best tool.
3. Use `registry.snapshot(restrict_to=[selected_tool])` to produce a restricted snapshot.
4. Generate the full tool-call grammar from the restricted snapshot and run the main constrained generation pass.

The architecture layers:
- **`tgirl.rerank`** — new module containing the `ToolRouter` class, routing grammar generation, and routing prompt generation.
- **`tgirl.grammar`** — add `generate_routing_grammar(snapshot)` to produce the minimal tool-name grammar.
- **`tgirl.sample.SamplingSession`** — integrate `ToolRouter` as an optional pre-pass before constrained generation.

The re-ranking pass reuses the same `GrammarState` protocol, `run_constrained_generation`, and OT transport infrastructure — no new inference primitives needed.

## 3. Architecture Impact

### Files/modules affected

| File | Change | Reason |
|------|--------|--------|
| `src/tgirl/rerank.py` | CREATE | New module: ToolRouter, routing grammar, routing prompt |
| `src/tgirl/grammar.py` | MODIFY | Add `generate_routing_grammar()` function |
| `src/tgirl/sample.py` | MODIFY | Integrate ToolRouter into SamplingSession |
| `src/tgirl/types.py` | MODIFY | Add `RerankConfig` and `RerankResult` models |
| `src/tgirl/__init__.py` | MODIFY | Export new public API |
| `tests/test_rerank.py` | CREATE | Unit tests for rerank module |
| `tests/test_sample_rerank.py` | CREATE | Integration tests for session + rerank |

### Data model changes

New types in `types.py`:
- `RerankConfig` — configuration for re-ranking (max_tokens, temperature, top_k fallback count)
- `RerankResult` — result of re-ranking (selected_tools, routing_grammar_text, tokens_used, latency_ms)

### API changes

New public API:
- `tgirl.rerank.ToolRouter` — stateless router that takes a snapshot and produces a restricted snapshot
- `tgirl.rerank.generate_routing_prompt(snapshot)` — generate the system prompt for tool routing
- `tgirl.grammar.generate_routing_grammar(snapshot)` — generate the minimal tool-name grammar

Modified API:
- `SamplingSession.__init__()` gains optional `rerank_config: RerankConfig | None` parameter
- `SamplingSession.run()` inserts re-ranking pass when `rerank_config` is not None

### Dependency additions

None. Reuses existing `outlines` + `torch` + `structlog`.

## 4. Acceptance Criteria

1. `generate_routing_grammar(snapshot)` produces a Lark grammar that accepts exactly the tool names in the snapshot, and no other strings.
2. `generate_routing_prompt(snapshot)` produces a system prompt listing tools with descriptions, instructing the model to select one.
3. `ToolRouter.route(request, snapshot, ...)` returns a `RerankResult` with the selected tool name(s).
4. When `rerank_config` is set, `SamplingSession` runs a re-ranking pass before each constrained generation cycle.
5. The restricted snapshot passed to constrained generation contains only the re-ranked tool(s).
6. Re-ranking is optional — `SamplingSession` works identically to current behavior when `rerank_config` is None.
7. Re-ranking telemetry (tokens used, latency, selected tool) is captured in `TelemetryRecord`.
8. `ToolRouter` respects quota constraints — exhausted tools are excluded from routing candidates.
9. All existing tests pass without modification.
10. Re-ranking with a single registered tool is a no-op (no routing pass needed).

## 5. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Re-ranking adds latency to every tool call | MEDIUM | LOW | Make it optional via config; sub-second on tested hardware |
| Routing grammar compilation cost (Outlines) | LOW | MEDIUM | Grammar is tiny (~7 alternatives); cache if needed |
| Re-ranking picks wrong tool, restricts grammar to it | MEDIUM | HIGH | Config option for top-K (restrict to top 2-3 instead of top 1) |
| Routing prompt quality varies by model | MEDIUM | LOW | Prompt is generated from registry metadata, not hand-tuned |
| `sample.py` is not yet on `main` | HIGH | HIGH | This feature branch must be rebased onto constrained-sampling-engine or merged after it |

## 6. Open Questions

1. **Branch dependency:** `sample.py` and `outlines_adapter.py` exist on `feature/constrained-sampling-engine` but not `main`. Should this branch be based off that branch instead of `main`? Or should we merge that PR first?
2. **Top-K vs top-1:** The experiment tested top-1 (perfect with grammar-constrained). Should we default to top-1 with a config option for top-K, or default to top-2 for safety?
3. **Routing prompt source:** Should the routing prompt reuse `generate_system_prompt()` from `instructions.py`, or should it have its own specialized format optimized for classification?

## 7. Out of Scope

- **Log-prob scoring as a re-ranking strategy** — experimentally proven ineffective (1/7). Not included.
- **Semi-structured re-ranking** — works (6/7) but fragile. Not included as a first-class strategy; users can implement custom routing via the `ToolRouter` protocol.
- **Multi-tool composition routing** — this feature routes to a single tool per cycle. Composition (threading, let-binding) is handled by the main grammar as before.
- **Model-specific prompt tuning** — the routing prompt is generated from registry metadata, not tuned per model.
- **Argument quality improvements** — re-ranking improves tool selection but not argument generation. The 3/7 executable rate is an orthogonal problem.
