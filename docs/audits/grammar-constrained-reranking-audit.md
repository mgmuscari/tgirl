# Security Audit: grammar-constrained-reranking

## Scope

**Examined:** All changes on `feature/grammar-constrained-reranking` vs `main` (~2,400 lines across 17 files). Primary focus on security-sensitive source code:
- `src/tgirl/rerank.py` (160 lines) — ToolRouter, routing logic, cache, quota filtering, output validation
- `src/tgirl/sample.py` (82 lines added) — rerank integration into SamplingSession.run(), telemetry wiring
- `src/tgirl/instructions.py` (155 lines) — prompt generation, routing prompt
- `src/tgirl/grammar.py` (91 lines changed) — routing grammar, type grammar injection
- `src/tgirl/types.py` (37 lines added) — RerankConfig, RerankResult, telemetry fields
- `src/tgirl/outlines_adapter.py` (111 lines) — GrammarState protocol implementation, error handling
- `src/tgirl/registry.py` (78 lines added) — register_type(), enrich(), type_grammars
- `src/tgirl/__init__.py` (6 lines added) — exports
- `examples/` (3 files) — e2e demo scripts

**NOT examined:** Thread safety / concurrent access, `compile.py` sandbox (separate audit exists), `transport.py` numerical correctness (separate audit exists), dependency CVEs (torch, outlines, llguidance), GPU-specific behavior, `bridge.py` MCP layer, llguidance internal error handling/recovery semantics.

## Methodology

Dual-agent team audit — Security Auditor and Skeptical Client operating as separate agents with direct peer messaging. Findings were challenged in real time, producing severity ratings that survived adversarial scrutiny.

## Findings Summary

| # | Severity (final) | Category | Description | Effort |
|---|-------------------|----------|-------------|--------|
| 1 | INFO | Resource Exhaustion | Routing context grows per cycle | XS |
| 2 | INFO | Configuration | Dead config fields (temperature, top_k) in RerankConfig | XS |
| 3 | LOW | Business Logic | Unhandled ValueError on quota exhaustion with reranking | XS |
| 4 | MEDIUM | Business Logic | Silent grammar state divergence on advance() error (pre-existing) | S |
| 5 | LOW | Input Validation | RerankConfig.max_tokens lacks gt=0 validation | XS |
| 6 | INFO | Business Logic | Shared forward_fn KV cache contamination | XS |
| 7 | INFO | Data Exposure | Rerank telemetry extends data exposure surface | XS |

**Post-challenge totals:** 0 CRITICAL, 0 HIGH, 1 MEDIUM, 2 LOW, 4 INFO

## Detailed Findings

### Finding 1: Routing context grows per cycle
**Initial Severity:** LOW — **Final Severity:** INFO
**Category:** Resource Exhaustion
**Affected Code:** `src/tgirl/rerank.py:109`
**Description:** Routing pass constructs context from routing prompt + full token_history, growing per cycle.
**Client Challenge:** Not routing-specific — main constrained pass does the same. Routing adds only 16 forward calls (negligible vs. 512 constrained, 4096 freeform). Session bounds prevent exploitation.
**Auditor Defense:** Acknowledged downgrade.
**Resolution:** Downgraded to INFO. Architectural property, not routing-introduced.
**Remediation:** Document as optimization opportunity (trailing window for routing context).
**Effort Estimate:** XS

### Finding 2: Dead config fields in RerankConfig
**Initial Severity:** INFO — **Final Severity:** INFO
**Category:** Configuration
**Affected Code:** `src/tgirl/types.py:257-258`
**Description:** `RerankConfig.temperature` and `RerankConfig.top_k` are defined but never referenced in `ToolRouter.route()`. False sense of configurability.
**Client Challenge:** Accepted at INFO.
**Resolution:** Code hygiene issue.
**Remediation:** Wire fields into routing pass or remove them.
**Effort Estimate:** XS

### Finding 3: Unhandled ValueError on quota exhaustion with reranking
**Initial Severity:** MEDIUM — **Final Severity:** LOW
**Category:** Business Logic
**Affected Code:** `src/tgirl/rerank.py:93-95`, `src/tgirl/sample.py:498`
**Description:** When all tools are quota-exhausted, `ToolRouter.route()` raises ValueError via `generate_routing_grammar()`. Unhandled in `SamplingSession.run()`, crashes session instead of returning partial results.
**Client Challenge:** Crash is semi-intentional (comment at line 93 confirms). Trigger is narrow (requires ALL tools to have explicit quotas AND all exhausted). No safety boundary crossed.
**Auditor Defense:** Acknowledged downgrade, noted the raise-via-side-effect pattern is fragile.
**Resolution:** Downgraded to LOW. Robustness improvement, not security fix.
**Remediation:** Explicit raise or sentinel return for empty tool sets. Catch in session loop.
**Effort Estimate:** XS

### Finding 4: Silent grammar state divergence on advance() error (PRE-EXISTING)
**Initial Severity:** MEDIUM — **Final Severity:** MEDIUM
**Category:** Business Logic / Input Validation
**Affected Code:** `src/tgirl/outlines_adapter.py:79-84`
**Description:** `LLGuidanceGrammarState.advance()` logs grammar errors as warnings but continues silently. If grammar state diverges, subsequent `get_valid_mask()` calls return incorrect masks, potentially breaking the core safety invariant.
**Proof of Concept:** Speculative — under current code, OT sets invalid tokens to -inf, softmax produces 0.0 probability, `multinomial` cannot select them. No concrete trigger path exists. Danger is that a future upstream change could be silently masked.
**Client Challenge:** PoC is speculative and finding is pre-existing, not introduced by reranking.
**Auditor Defense:** Defended MEDIUM on defense-in-depth grounds. The warn-and-continue pattern is dangerous if upstream invariants break. The safety invariant deserves fail-fast protection.
**Resolution:** MEDIUM maintained. Reclassified as pre-existing finding noted during audit.
**Remediation:** Raise exception or propagate error when `get_error()` returns non-None.
**Effort Estimate:** S

### Finding 5: RerankConfig.max_tokens lacks validation
**Initial Severity:** LOW — **Final Severity:** LOW
**Category:** Input Validation
**Affected Code:** `src/tgirl/types.py:252-259`
**Description:** `max_tokens` accepts zero/negative values, causing empty generation and ValueError.
**Client Challenge:** Accepted at LOW.
**Resolution:** Developer-facing config gap, same pattern as prior audit Finding 8.
**Remediation:** Add `Field(gt=0)` constraint.
**Effort Estimate:** XS

### Finding 6: Shared forward_fn KV cache contamination
**Initial Severity:** MEDIUM — **Final Severity:** INFO
**Category:** Business Logic
**Affected Code:** `src/tgirl/rerank.py:127-128`, `src/tgirl/sample.py:414`
**Description:** ToolRouter shares `forward_fn` with SamplingSession. If `forward_fn` has KV cache state, routing calls with different context prefix could contaminate cache.
**Client Challenge:** `forward_fn` contract is already "full sequence in, logits out" across the entire codebase. Routing follows the same contract. A stateful `forward_fn` would break without reranking too. PoC proves broken caller, not broken library.
**Auditor Defense:** Acknowledged downgrade.
**Resolution:** Downgraded to INFO. Documentation gap about `forward_fn` statelessness contract.
**Remediation:** Document statelessness requirement in `forward_fn` docstring/type alias.
**Effort Estimate:** XS

### Finding 7: Rerank telemetry extends data exposure surface
**Initial Severity:** INFO — **Final Severity:** INFO
**Category:** Data Exposure
**Affected Code:** `src/tgirl/types.py:244-246`
**Description:** New telemetry fields expose routing decisions and performance. Incremental extension of prior audit Finding 6.
**Client Challenge:** Accepted at INFO.
**Resolution:** System metadata at trusted boundary.
**Remediation:** Include in telemetry sensitivity documentation.
**Effort Estimate:** XS

## What This Audit Did NOT Find

- No exploitable security vulnerabilities where external or untrusted input can bypass grammar safety guarantees
- No authentication or authorization bypass paths (module has no auth surface)
- No injection vectors in routing prompt — grammar constraints make prompt injection in tool descriptions ineffective (output is constrained to valid tool names regardless of prompt content)
- No cryptographic issues (module uses no cryptography)
- No dependency vulnerabilities examined

**Key observation:** The grammar-constrained architecture provides excellent defense against prompt injection in the routing prompt. Even if tool descriptions contain adversarial text, the routing grammar constrains output to valid tool names only. This validates the core design claim.

## Remediation Priority

1. **F4** (MEDIUM, pre-existing, S) — Fail-fast on grammar advance error (defense-in-depth for core safety invariant)
2. **F3** (LOW, XS) — Explicit error handling for empty tool sets in routing
3. **F5** (LOW, XS) — Add RerankConfig field validation
4. **F2** (INFO, XS) — Wire or remove dead config fields
5. **F1, F6, F7** (INFO, XS each) — Documentation improvements
