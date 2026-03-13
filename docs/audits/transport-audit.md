# Security Audit: transport

## Scope

**Examined:**
- `src/tgirl/transport.py` (324 lines) — optimal transport logit redistribution module
- `tests/test_transport.py` (667 lines) — unit tests
- `tests/test_integration_transport.py` (166 lines) — integration tests
- `src/tgirl/__init__.py` — transport exports

**Not examined:**
1. GPU/CUDA-specific behavior (no GPU available in audit environment)
2. Concurrency/thread safety
3. Dependency CVEs (torch, pydantic, structlog versions)
4. Batched 2D logit inputs
5. Autograd graph retention/leakage

## Methodology

Dual-agent team audit — Security Auditor and Skeptical Client operating as separate agents with direct peer messaging. Findings were challenged in real time, producing severity ratings that survived adversarial scrutiny. Each finding followed a fixed protocol: auditor reports → client challenges → auditor defends or acknowledges → finding closed with final severity.

## Findings Summary

| # | Severity (final) | Category | Description | Effort |
|---|------------------|----------|-------------|--------|
| F3 | HIGH | Numerical Correctness | Valid tokens silently removed by -inf | XS |
| F1 | MEDIUM | Input Validation | epsilon=0 causes NaN logit propagation | XS |
| F2 | LOW | Denial of Service | OOM via unbounded cost submatrix | S |
| F4 | LOW | Configuration | No bounds on TransportConfig fields | XS |
| F6 | INFO | Error Quality | Raw PyTorch errors on shape/dtype mismatch | XS |

2 findings withdrawn (F5 merged into F4, F7 dismissed as expected behavior).

## Detailed Findings

### Finding 3: Valid tokens silently removed by -inf

**Initial Severity:** HIGH
**Final Severity:** HIGH
**Category:** Numerical Correctness
**Affected Code:** `src/tgirl/transport.py:284-287` (target_capacity normalization), `src/tgirl/transport.py:148-149` (torch.log of zero mass)
**Description:** When logit differences exceed ~104 (float32), softmax underflows to 0.0 for some valid tokens. After normalization into `target_capacity`, these zeros produce `-inf` via `torch.log()`, propagating through Sinkhorn. Valid tokens receive `-inf` in output — silently removed from the valid set with no crash or downstream detection.
**Proof of Concept:** `logits=[1000, -1000, ...]` with both tokens marked valid → token with logit=-1000 gets `-inf` in output despite being grammar-valid.
**Client Challenge:** Extreme logit values unlikely in practice; affected tokens had vanishing probability anyway; should be MEDIUM.
**Auditor Defense:** (1) Logit spans of 100+ occur in real LLMs, especially early tokens and with temperature scaling. (2) This is a silent, undetectable violation — the sampler cannot distinguish a transport-zeroed valid token from a correctly-masked invalid token. (3) Violates the project's core "safety by construction" invariant. (4) Fix is trivial.
**Resolution:** Severity maintained at HIGH. Client accepted the "silent undetectable violation" argument.
**Remediation:** `torch.clamp(target_capacity, min=1e-30)` and same for `source_mass` before normalization.
**Effort Estimate:** XS

### Finding 1: epsilon=0 causes NaN logit propagation

**Initial Severity:** HIGH
**Final Severity:** MEDIUM
**Category:** Input Validation / Numerical Correctness
**Affected Code:** `src/tgirl/transport.py:142` (`log_kernel = -cost_matrix / epsilon`), `src/tgirl/transport.py:22-31` (TransportConfig)
**Description:** `epsilon=0.0` causes division by zero in `_sinkhorn_log_domain`, producing NaN logits. Negative epsilon also accepted but produces meaningless results. No validation on epsilon being positive.
**Proof of Concept:** `TransportConfig(epsilon=0.0)` → `redistribute_logits()` returns NaN logits.
**Client Challenge:** Internal API with sensible default (0.1). Requires deliberate misconfiguration. NaN caught by downstream `torch.multinomial` (crashes, doesn't silently pass through).
**Auditor Defense:** Accepted downgrade. Verified multinomial does crash on NaN, though error message is opaque.
**Resolution:** Downgraded from HIGH to MEDIUM. Internal API boundary and downstream crash detection reduce severity.
**Remediation:** Add `gt=0` constraint to `epsilon` field via Pydantic `Field`.
**Effort Estimate:** XS

### Finding 2: OOM via unbounded cost submatrix

**Initial Severity:** MEDIUM
**Final Severity:** LOW
**Category:** Denial of Service
**Affected Code:** `src/tgirl/transport.py:95-114` (`_compute_cost_submatrix`)
**Description:** Cost submatrix is `O(n_invalid * n_valid)` with no cap. For 100k vocab with 30% valid, this allocates ~8.4 GB. Real LLM vocabs (32k-128k) make this theoretically reachable. OOM kills the process with no recovery.
**Proof of Concept:** Calculation-based: 50k invalid × 50k valid × 4 bytes = 9.3 GB.
**Client Challenge:** Grammar constraints produce small valid sets in practice (dozens to thousands, not 30% of vocab). Bypass catches >50% valid ratio. Should be LOW.
**Auditor Defense:** Initially defended MEDIUM, but after reviewing TGIRL.md spec confirmed grammar-constrained decoding produces small valid sets. Accepted downgrade to LOW.
**Resolution:** Downgraded from MEDIUM to LOW. Grammar constraints make large valid sets unrealistic in practice.
**Remediation:** Add size-based bypass condition (e.g., `n_invalid * n_valid > max_problem_size`) that falls back to standard masking.
**Effort Estimate:** S

### Finding 4: No bounds on TransportConfig fields

**Initial Severity:** LOW (includes merged F5: DoS via max_iterations)
**Final Severity:** LOW
**Category:** Configuration / Input Validation
**Affected Code:** `src/tgirl/transport.py:22-31` (TransportConfig)
**Description:** No field-level validation beyond type checking. Accepts negative `max_iterations`, negative thresholds, out-of-range ratios. `valid_ratio_threshold=-0.5` silently degrades all transport to standard masking. `max_iterations=1000000` demonstrated 155s CPU exhaustion on 100 tokens (merged from F5).
**Client Challenge:** Programmer errors with nonsensical inputs. F5 was a duplicate repackaged as DoS.
**Auditor Defense:** Accepted F5 merge. F4 stands at LOW.
**Resolution:** F5 merged into F4. LOW severity.
**Remediation:** Add Pydantic `Field` constraints: `epsilon=Field(gt=0)`, `max_iterations=Field(ge=1, le=1000)`, `convergence_threshold=Field(gt=0)`, `valid_ratio_threshold=Field(ge=0, le=1)`, `invalid_mass_threshold=Field(ge=0, le=1)`.
**Effort Estimate:** XS

### Finding 6: Poor error messages on shape/dtype mismatch

**Initial Severity:** INFO
**Final Severity:** INFO
**Category:** Error Quality
**Affected Code:** `src/tgirl/transport.py:222-324` (redistribute_logits)
**Description:** Shape/dtype mismatches raise raw PyTorch `IndexError`/`RuntimeError` with no domain context.
**Client Challenge:** Not a security finding. PyTorch errors are sufficient. CLAUDE.md says validate at system boundaries only.
**Auditor Defense:** Accepted INFO.
**Resolution:** INFO. Optional improvement if already editing the file.
**Remediation:** Add shape/dtype assertions at top of `redistribute_logits`.
**Effort Estimate:** XS

### Withdrawn Findings

**Finding 5 (DoS via uncapped max_iterations):** Merged into Finding 4 as supporting evidence for config validation.

**Finding 7 (Debug log leakage):** Client challenged — debug logs for local inference are expected operational metadata. Log access implies system access. Dismissed.

## What This Audit Did NOT Find

- No exploitable remote vulnerabilities (module is an internal library, not a service)
- No injection vectors (operates on tensors, not strings)
- No authentication/authorization issues (not applicable)
- No data persistence or state leakage between calls
- No dependency on unsafe deserialization

## Positive Observations

- Zero coupling verified (no tgirl imports in transport.py) — confirmed via grep and existing AST test
- Input tensors not mutated (clone on entry)
- Mass conservation holds under normal conditions (verified across multiple seeds)
- Standard masking bypass path is correct and non-mutating
- Sinkhorn marginal conservation holds (verified via Hypothesis property test)
- Test coverage is thorough (667 lines unit + 166 lines integration = 833 lines for 324 lines of implementation)

## Remediation Priority

1. **F3** (HIGH, XS) — Clamp target_capacity/source_mass to min=1e-30. Silent safety invariant violation, trivial fix. **Fix first.**
2. **F1+F4** (MEDIUM+LOW, XS) — Add Pydantic Field constraints to TransportConfig. Single commit. **Fix second.**
3. **F2** (LOW, S) — Add size-based bypass to _compute_cost_submatrix. **Fix third.**
4. **F6** (INFO, XS) — Optional shape/dtype assertions. Bundle with other changes if convenient.

Total remediation effort: 2× XS + 1× S.
