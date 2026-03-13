# Security Audit: constrained-sampling-engine

## Scope

**Examined:** All changes on `feature/constrained-sampling-engine` vs `main` (2,640 lines added across 9 files). Primary focus on security-sensitive code:
- `src/tgirl/sample.py` (624 lines) — sampling loop, hook protocol, logit processing, delimiter detection, session orchestration, cross-cycle quota tracking
- `src/tgirl/types.py` (45 lines added) — SessionConfig, ModelIntervention models

**NOT examined:** `compile.py` sandbox (separate audit exists), `grammar.py` quota enforcement at grammar level, `transport.py` numerical guarantees (separate audit exists), thread safety, `bridge.py`, dependency CVEs, GPU-specific behavior.

## Methodology

Dual-agent team audit — Security Auditor and Skeptical Client operating as separate agents with direct peer messaging. Findings were challenged in real time, producing severity ratings that survived adversarial scrutiny. The auditor hunted for vulnerabilities; the client challenged each finding for evidence quality, realistic exploitability, and severity calibration.

## Findings Summary

| # | Initial Severity | Final Severity | Category | Description | Effort |
|---|------------------|----------------|----------|-------------|--------|
| 1 | HIGH | INFO | Business Logic | Hook logit_bias can override model distribution | XS (doc) |
| 2 | MEDIUM | LOW | Numerical Stability | Zero valid tokens causes NaN in constrained path | XS |
| 3 | MEDIUM | LOW | Numerical Stability | Freeform path missing zero-probability fallback | XS |
| 4 | HIGH | MEDIUM | Business Logic | Regex quota counting can undercount | XS |
| 5 | LOW | INFO | Design | Delimiter spoofing via freeform generation | XS (doc) |
| 6 | INFO | INFO | Data Exposure | Telemetry contains raw tool outputs | XS (doc) |
| 7 | MEDIUM | MEDIUM | Resource Exhaustion | No timeout in constrained generation loop | S |
| 8 | LOW | LOW | Input Validation | SessionConfig lacks field validation | XS |
| 9 | LOW | INFO | Design | Result injection may influence delimiter generation | - |

**Post-challenge totals:** 0 CRITICAL, 0 HIGH, 2 MEDIUM, 3 LOW, 4 INFO

## Detailed Findings

### Finding 1: Hook logit_bias can override model distribution
**Initial Severity:** HIGH
**Final Severity:** INFO
**Category:** Business Logic / Authorization Bypass
**Affected Code:** `src/tgirl/sample.py:139-142`, `src/tgirl/sample.py:278-282`
**Description:** A malicious InferenceHook can supply extreme logit_bias values that override the model's distribution entirely, selecting whichever grammar-valid token the hook wants.
**Proof of Concept:** Hook returning `logit_bias={target_id: 1000.0}` forces token selection.
**Client Challenge:** Hooks are instantiated by the application developer (trusted boundary), not by model output or external input. Grammar constraints are NOT bypassed. The PoC only demonstrates selecting among grammar-valid tokens, which is the intended purpose of hooks.
**Auditor Defense:** Conceded downgrade. Grammar constraints preserved; no actual security boundary crossed.
**Resolution:** Downgraded to INFO. Hooks are a trusted extension point by design.
**Remediation:** Document that hooks are trusted code with full control over token selection within grammar constraints.
**Effort Estimate:** XS

### Finding 2: Zero valid tokens causes NaN in constrained path
**Initial Severity:** MEDIUM
**Final Severity:** LOW
**Category:** Numerical Stability
**Affected Code:** `src/tgirl/sample.py:318`
**Description:** If grammar state returns an all-False valid mask, the fallback probability normalization divides by zero, producing NaN.
**Client Challenge:** Precondition requires a fundamentally broken grammar state. tgirl's grammars are generated from registry snapshots. Transport layer handles this upstream via `forced_decode` bypass.
**Auditor Defense:** Did not rebut.
**Resolution:** Downgraded to LOW. Add guard for developer experience, not security.
**Remediation:** Add early return or descriptive error when `valid_mask.sum() == 0`.
**Effort Estimate:** XS

### Finding 3: Freeform path missing zero-probability fallback
**Initial Severity:** MEDIUM
**Final Severity:** LOW
**Category:** Numerical Stability
**Affected Code:** `src/tgirl/sample.py:448-452`
**Description:** Freeform sampling has no fallback for zero probability sum, unlike the constrained path. Crashes with RuntimeError on degenerate logits.
**Client Challenge:** Precondition requires `forward_fn` returning all -inf logits (broken model). `torch.softmax` on any finite tensor produces a valid distribution.
**Auditor Defense:** Did not rebut.
**Resolution:** Downgraded to LOW. Fix asymmetry with constrained path for consistency.
**Remediation:** Add uniform fallback matching constrained path.
**Effort Estimate:** XS

### Finding 4: Regex quota counting can undercount
**Initial Severity:** HIGH
**Final Severity:** MEDIUM
**Category:** Business Logic
**Affected Code:** `src/tgirl/sample.py:553`
**Description:** Regex-based `_count_tool_invocations` can undercount in edge cases (comments, aliasing).
**Client Challenge:** Both attack scenarios are blocked by other layers. Aliasing (`setv`) rejected by Hy AST static analyzer. Comments impossible in grammar-constrained output. Grammar-level quota enforcement is the primary defense.
**Auditor Defense:** Defended MEDIUM: regex counting is the session-level quota enforcement mechanism. While grammar-level enforcement is the primary defense, the secondary layer should be correct. A contract gap in the counting function weakens defense-in-depth. Not HIGH (grammar prevents exploitation), but stronger than LOW (it's the designated counting mechanism failing at its job).
**Resolution:** Settled at MEDIUM. Grammar prevents exploitation but AST-based counting should replace regex for correctness of the secondary defense layer.
**Remediation:** Replace regex counting with AST-based counting via existing compile pipeline.
**Effort Estimate:** XS

### Finding 5: Delimiter spoofing via freeform generation
**Initial Severity:** LOW
**Final Severity:** INFO
**Category:** Design
**Affected Code:** `src/tgirl/sample.py` (DelimiterDetector)
**Description:** Model can accidentally emit the tool call delimiter during freeform generation.
**Client Challenge:** Inherent property of delimiter-based mode switching (shared by ChatML, etc.). Delimiter is configurable. Even if triggered, grammar constrains output to valid tool calls.
**Auditor Defense:** Did not rebut.
**Resolution:** Downgraded to INFO. Design property, not vulnerability.
**Remediation:** Document delimiter selection guidance.
**Effort Estimate:** XS

### Finding 6: Telemetry contains raw tool outputs
**Initial Severity:** INFO
**Final Severity:** INFO
**Category:** Data Exposure
**Affected Code:** `src/tgirl/sample.py` (TelemetryRecord population)
**Description:** Telemetry records include raw tool execution results and Hy source, potentially exposing sensitive data.
**Client Challenge:** Agreed with INFO. Every telemetry system has this property. Caller controls what they do with SamplingResult.
**Resolution:** INFO confirmed. Documentation only.
**Remediation:** Document telemetry data sensitivity in API docs.
**Effort Estimate:** XS

### Finding 7: No timeout in constrained generation loop
**Initial Severity:** MEDIUM
**Final Severity:** MEDIUM
**Category:** Resource Exhaustion
**Affected Code:** `src/tgirl/sample.py` (run_constrained_generation)
**Description:** `run_constrained_generation` has no timeout check; only bounded by `max_tokens`. Session timeout only enforced in freeform loop.
**Client Challenge:** Constrained loop bounded by `max_tokens=512` default. Each OT call bounded by `max_iterations=20`. Worst case is slow but not unbounded.
**Auditor Defense:** Defended MEDIUM: the session-level timeout exists and is enforced in the freeform loop but NOT in the constrained loop — this is an asymmetry in a security-relevant control. While max_tokens bounds the loop, the absence of timeout in constrained mode is a contract gap. The freeform loop having it makes the omission more notable, not less.
**Resolution:** Settled at MEDIUM. Contract gap — timeout control exists at session level but is not propagated to constrained generation.
**Remediation:** Add optional timeout parameter to `run_constrained_generation` for consistency with session-level timeout.
**Effort Estimate:** S

### Finding 8: SessionConfig lacks field validation
**Initial Severity:** LOW
**Final Severity:** LOW
**Category:** Input Validation
**Affected Code:** `src/tgirl/types.py` (SessionConfig)
**Description:** SessionConfig fields like `max_tool_cycles`, `session_timeout`, temperatures accept nonsensical values (negative, zero) without validation. TransportConfig has Field constraints; SessionConfig does not.
**Client Challenge:** No challenge. Correctly scoped and rated.
**Resolution:** LOW confirmed. Consistency improvement.
**Remediation:** Add `Field(gt=0)` / `Field(ge=0)` constraints matching TransportConfig pattern.
**Effort Estimate:** XS

### Finding 9: Result injection may influence delimiter generation
**Initial Severity:** LOW
**Final Severity:** INFO
**Category:** Design
**Affected Code:** `src/tgirl/sample.py` (SamplingSession.run)
**Description:** Tool result tokens injected into context may influence the model to re-emit the tool call delimiter.
**Client Challenge:** Result tokens are NOT fed to the delimiter detector. Influence is indirect (model is influenced by context, which is how language models work). Even if delimiter produced, grammar constraints hold.
**Auditor Defense:** Did not rebut.
**Resolution:** Downgraded to INFO. Not a vulnerability.
**Remediation:** None required.
**Effort Estimate:** -

## What This Audit Did NOT Find

- No exploitable security vulnerabilities where external or untrusted input can bypass grammar safety guarantees
- No authentication or authorization bypass paths (module has no auth surface)
- No injection vectors (Hy compilation and sandbox are in compile.py, audited separately)
- No cryptographic issues (module uses no cryptography)
- No dependency vulnerabilities examined

## Remediation Priority

1. **Finding 4** (MEDIUM) — Replace regex counting with AST-based counting (XS, defense-in-depth correctness)
2. **Finding 7** (MEDIUM) — Add timeout param to run_constrained_generation (S, contract gap)
3. **Finding 8** (LOW) — Add SessionConfig field validation (XS, consistency with TransportConfig)
4. **Finding 2** (LOW) — Add zero-valid-tokens guard in constrained path (XS, defense-in-depth)
5. **Finding 3** (LOW) — Add zero-prob fallback in freeform path (XS, consistency)
6. **Findings 1, 5, 6** (INFO) — Documentation improvements (XS each)

All remediations are robustness/hygiene improvements, not security-critical fixes.
