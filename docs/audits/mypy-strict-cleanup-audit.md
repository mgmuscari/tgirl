# Security Audit: mypy-strict-cleanup

## Scope

**Examined:**
- `git diff main...HEAD` on branch `feature/mypy-strict-cleanup` — 19 commits, ~14 files in `src/`, ~1,250 LoC.
- Deep focus per the pre-audit brief:
  - Task 12 architectural change: `ToolRouter` now takes `mlx_grammar_guide_factory` parameter; dispatch in `ToolRouter.route()` based on backend; cross-framework fallback at `sample_mlx.py:506-507` deleted.
  - Task 13 `# type: ignore[misc]` at `compile.py:439` for `RestrictedPython.RestrictingNodeTransformer` dynamic base.
  - Task 13 `PromptFormatter.format_messages` Protocol widening to accept `**kwargs: object`.
  - Task 9 `bridge.py:538` field access change `.text` → `.output_text`.
  - New Protocols at Task 5 (`EstradiolControllerProto`, `ConfidenceMonitorProto`), Task 6 (`SteerableForwardFn`), Task 7 (`TokenizerProto`).
  - Behavior change in `state_machine.py`: non-int/str `token_id` now returns no-op `TransitionDecision` instead of raising `TypeError`.
  - Five new `assert` statements in `sample.py` / `sample_mlx.py` used as mypy type-narrowing hints.

**NOT examined (explicit limitations):**
- `compile.py:800` sandboxed bytecode execution site — unchanged by this PR; pre-existing sandbox core, out of scope.
- Supply-chain audit of `RestrictedPython` / `llguidance` / `outlines` / `POT` — no dependency version bumps in diff.
- Performance regression scan.
- Non-diff sections of `calibrate.py` / `serve.py` / `cli.py`.
- Full `tests/test_compile.py` rerun (verified sandbox tests weren't weakened by inspection only, not by executing pytest).
- `compile.py:800` and related sandbox core are pre-existing audit debt on `main`, not blockers for this PR.

## Methodology

Dual-agent team audit — Security Auditor (hard stance) and Skeptical Client (hard stance) operating as separate agents with direct peer messaging. Each finding followed the fixed protocol: auditor reports → client challenges → auditor defends or acknowledges → finding CLOSES at final severity. No oscillation, one round per finding. Findings survive only if they stand up to adversarial calibration pressure.

## Findings Summary

| # | Final Severity | Category | Description | Effort |
|---|---|---|---|---|
| 1 | INFO | Business logic | Task 12 `ToolRouter` backend-aware factory dispatch — actually improves security posture | 0 |
| 2 | INFO | Sandbox | `compile.py:439` `# type: ignore[misc]` on `RestrictedPython` dynamic base is stub-gap only; no runtime effect | 0 |
| 3 | INFO | Input validation | `PromptFormatter.format_messages` widened to `**kwargs: object`; Jinja2 `ImmutableSandboxedEnvironment` remains the boundary | 0 |
| 4 | INFO | Data exposure | `bridge.py:538` `.text`→`.output_text` is a latent-bug fix; pre-fix worst case was DoS (AttributeError), not data leak | 0 |
| 5 | INFO (↓ from LOW) | Type safety | `@runtime_checkable` Protocols only verify method existence; no production call site uses `isinstance` as a trust gate | XS |
| 6 | INFO (↓ from LOW) | Supply chain | Malicious tokenizer via `TokenizerProto` — pre-existing concern on main; diff neither introduces nor worsens | N/A |
| 7 | LOW | Defensive programming | Five production `assert` statements stripped under `python -O` | S |

**Aggregate:** 0 CRITICAL, 0 HIGH, 0 MEDIUM, 6 INFO, 1 LOW.

## Detailed Findings

### Finding 1: MLX factory dispatch improves posture
**Initial Severity:** INFO
**Final Severity:** INFO
**Category:** Business logic
**Affected Code:** `src/tgirl/rerank.py:35-42, 130-136`; `src/tgirl/sample.py:760, 828`; `src/tgirl/sample_mlx.py:508-521`
**Description:** Task 12 added `mlx_grammar_guide_factory` to `ToolRouter.__init__`, wired through `SamplingSession.__init__`, and dispatched in `ToolRouter.route()` on `backend == "mlx"`. The old cross-framework fallback that coerced a torch mask into an MLX array was deleted and replaced with a fail-loud `TypeError`.
**Proof of Concept:** All three potential misuse paths were tested by the auditor:
- Missing MLX factory: `TypeError` raised fail-loud.
- Swapped factory (torch factory where MLX expected): `TypeError` on `grammar_state.get_valid_mask_mx`.
- Spoofed `get_valid_mask_mx` returning a torch tensor: downstream `mx` ops reject the tensor, no silent coercion.
**Client Challenge:** None — accepted.
**Auditor Defense:** N/A.
**Resolution:** Net security-positive change. Eliminates CLAUDE.md 2026-03-14 gotcha (no cross-framework conversions) and replaces it with fail-loud contract enforced by a new regression test.
**Remediation:** None.
**Effort Estimate:** 0.

### Finding 2: `compile.py:439` `# type: ignore[misc]` is stub-gap
**Initial Severity:** INFO
**Final Severity:** INFO
**Category:** Sandbox
**Affected Code:** `src/tgirl/compile.py:439`
**Description:** `class _TgirlNodeTransformer(RestrictingNodeTransformer):  # type: ignore[misc]` silences mypy's "Class cannot subclass Any" warning — caused by `RestrictedPython` shipping without `.pyi` stubs. The ignore does NOT suppress protocol/override/signature mismatch errors, which mypy is still catching in adjacent code.
**Proof of Concept:** N/A — stub-gap silencer, no runtime behavior change. All sandbox visitors (`visit_Import`/`ImportFrom`/`Global`/`Nonlocal`, `check_name` `_hy_*` prefix, `visit_Attribute` dunder-rejection) are unchanged by this PR and continue to enforce sandbox invariants.
**Client Challenge:** Noted that `pyproject.toml` already sets `ignore_missing_imports` for `RestrictedPython` per Task 1; the local `[misc]` ignore is still required for the dynamic-base-class warning and is structurally necessary until upstream ships `.pyi` stubs.
**Auditor Defense:** Accepted — ignore is structural, not a sandbox weakener.
**Resolution:** Approved. No action required until upstream stubs are available.
**Remediation:** Already applied (pyproject override); local ignore is structural.
**Effort Estimate:** 0.

### Finding 3: `PromptFormatter.format_messages` widened
**Initial Severity:** INFO
**Final Severity:** INFO
**Category:** Input validation
**Affected Code:** `src/tgirl/types.py:340`; `src/tgirl/format.py:38, 57, 67`
**Description:** Protocol widened to accept `**kwargs: object`. `ChatTemplateFormatter` pops `add_generation_prompt` with `bool()` coercion, forwards remainder to `tokenizer.apply_chat_template`. `PlainFormatter` does `del kwargs` to drop arbitrary keys.
**Proof of Concept:** No shell-injection path exists. Jinja2's `ImmutableSandboxedEnvironment` (which chat templates use) prevents `import` / attribute-escape patterns; kwargs become template *variables*, not expressions, so even a maliciously-constructed value can't execute Python code through the template engine.
**Client Challenge:** Accepted. PEP 692 TypedDict-unpack hardening would tighten the spec further but is nice-to-have, not this PR's scope.
**Auditor Defense:** Agreed.
**Resolution:** Sandbox boundary (Jinja2) holds; new `**kwargs` arg doesn't create a new sink.
**Remediation:** None required. Future hardening via TypedDict `**kwargs: Unpack[T]` is backlog.
**Effort Estimate:** 0.

### Finding 4: `bridge.py:538` `.output_text` field access
**Initial Severity:** INFO
**Final Severity:** INFO
**Category:** Data exposure
**Affected Code:** `src/tgirl/bridge.py:539-543`
**Description:** The MCP handler previously accessed `result.text` on `SamplingResult`, which does not have that attribute — `SamplingResult` exposes `.output_text`. Task 9 fixed it. Worst case pre-fix was `AttributeError` at runtime (a DoS, masked in tests by `MagicMock` auto-synthesis), not a data leak.
**Proof of Concept:** Model-generated string is returned as MCP text content. No shell/eval/SQL/filesystem sink in the diff. Pre-fix: `AttributeError`; the attribute didn't exist, so nothing was leaking.
**Client Challenge:** Accepted — new regression test `test_expose_as_mcp_uses_output_text_field_on_real_sampling_result` constructs a real `SamplingResult` end-to-end through FastMCP `call_tool`, locking the field contract.
**Auditor Defense:** Agreed.
**Resolution:** Latent bug fix, not a new vulnerability.
**Remediation:** Already applied + regression test.
**Effort Estimate:** 0.

### Finding 5: `@runtime_checkable` Protocols don't enforce contracts
**Initial Severity:** LOW
**Final Severity:** INFO (downgraded)
**Category:** Type safety / trust boundary
**Affected Code:** `state_machine.py:351-368`; `estradiol.py:49-65`; `format.py:12-29`; `sample_mlx.py:29-50`; `types.py:328-342`
**Description:** `@runtime_checkable` Protocols verify method *existence*, not method *signatures* or *semantics*. A malicious duck-typed object could pass `isinstance(obj, Proto)` without actually honoring the contract.
**Proof of Concept:** An exploit requires a production call site that uses `isinstance(x, Proto)` as a trust gate. A grep of the source tree confirms no production isinstance-as-gate usage exists on the new Protocols. Only tests-only usage found. Production consumers call methods directly and would fail-loud on the first bad method.
**Client Challenge:** Downgrade to INFO. The auditor's own prose said "no trust gate exists," "not an untrusted-input boundary," and "PoC possible ONLY IF a gate that doesn't exist is introduced." LOW implies current residual risk; there is none given no production isinstance-as-gate usage.
**Auditor Defense:** Conceded. Downgraded.
**Resolution:** INFO — optional docstring on each Protocol noting "structural existence checks only; do not use as a trust gate" is the appropriate remediation shape.
**Remediation:** Optional docstring note in each Protocol.
**Effort Estimate:** XS.

### Finding 6: Malicious tokenizer via `TokenizerProto`
**Initial Severity:** LOW
**Final Severity:** INFO (downgraded)
**Category:** Supply chain
**Affected Code:** `src/tgirl/format.py:32-83` (`TokenizerProto` + `ChatTemplateFormatter`); `src/tgirl/serve.py:85-91` (mlx_lm.load path)
**Description:** A malicious tokenizer's Jinja2 `chat_template` could exfiltrate or manipulate prompts (not execute arbitrary Python due to `ImmutableSandboxedEnvironment`). Example template: `{{ 'EXFIL:' + messages[0].content }}` would prepend an exfil marker to the rendered prompt visible to downstream processing.
**Proof of Concept:** Requires an attacker-controlled model artifact loaded via `mlx_lm.load`. This risk is identical on `main`; this PR does not change the surface area.
**Client Challenge:** Downgrade to INFO. Auditor's own prose said "PR does NOT introduce or worsen," "behavior unchanged," "not this PR's problem," and marked effort as "N/A (out of scope)." A PR audit should not flag a pre-existing concern identical on main at LOW.
**Auditor Defense:** Conceded. Belongs in `SECURITY.md` as environment-hardening guidance, not in this PR's findings.
**Resolution:** INFO — document in `SECURITY.md`: "Load model artifacts only from trusted sources."
**Remediation:** Out of scope for this PR; add to `SECURITY.md` backlog.
**Effort Estimate:** N/A for this PR.

### Finding 7: Production asserts stripped under `python -O`
**Initial Severity:** LOW
**Final Severity:** LOW
**Category:** Defensive programming
**Affected Code:**
- `src/tgirl/sample.py:1015-1018` (`_empty_mask`, `_compute_signal`)
- `src/tgirl/sample.py:1112` (`self._router`)
- `src/tgirl/sample.py:1210` (`isinstance(hook, ModMatrixHook)`)
- `src/tgirl/sample_mlx.py:525-526` (`estradiol_alphas`, `estradiol_deltas`)
**Description:** Five `assert` statements were introduced during Tasks 6, 8, 10, 11, 12 as mypy type-narrowing hints. Under `python -O` these are stripped at bytecode compile time. On any deployment running `python -O`, the narrowing disappears and downstream code would hit `AttributeError`/`TypeError` on the None dereference — fail-loud, but with less clear diagnostics than an explicit raise.
**Proof of Concept:** Run under `python -O`: `_empty_mask` assert → `TypeError` at the subsequent tensor op. `_router` assert is redundant with the `rerank_active` predicate gate, so stripping it is low impact. `isinstance(hook, ModMatrixHook)` would let the code proceed to `hook.config` on a spoofed `__qualname__`/`__module__` alien class — but this is a pre-existing design trust in the `_is_mod_matrix_hook` predicate, not a new weakness. `estradiol_alphas`/`_deltas` asserts → `AttributeError` at list append if stripped. All failures remain fail-loud.
**Client Challenge:** Severity accepted — `-O` is a real runtime mode, not a hypothetical gate; the idiom is known-sketchy; remediation is concrete. Client added a light probe about silent-corruption clarity on the `isinstance(hook, ModMatrixHook)` path, which the auditor addressed (still fail-loud under `-O`, just less informative).
**Auditor Defense:** Standing at LOW; all sites remain fail-loud, but replacing with explicit `if x is None: raise RuntimeError(...)` would improve diagnostic quality and be `-O`-safe.
**Resolution:** Optional hardening for `-O` compatibility. Not merge-blocking.
**Remediation:** Replace the 5 `assert` sites with explicit `if` + `raise` constructs.
**Effort Estimate:** S.

## What This Audit Did NOT Find

- **No authentication/authorization bypasses.** No new auth surfaces introduced by this diff.
- **No new untrusted-input sinks.** `PromptFormatter` kwargs become template variables (Jinja2 sandbox), not expression targets.
- **No new dynamic code execution call sites.** The only pre-existing such site (sandboxed bytecode execution at `compile.py:800`) is unchanged.
- **No tests weakened or deleted.** All changes to test files are additions (5 new tests) or scope tightening — verified by inspection of the diff.
- **No dependency version bumps.** `pyproject.toml` changes are mypy config only.
- **No new API surface exposing internal state** beyond the already-public `SamplingResult.output_text` that `bridge.py` now correctly reads.
- **No hidden `# type: ignore` masking real vulnerabilities.** All 4 new `# type: ignore[rule]` entries have been inspected and carry specific rule specifiers + one-line rationale comments pointing at upstream stub gaps (`mlx_lm.load`, `llguidance.mlx`, `RestrictedPython`).

## Remediation Priority

**Blocking for merge:** None.

**Recommended non-blocking follow-ups:**

1. **Finding #7 (LOW, effort S, optional):** Replace the 5 `assert` sites with explicit `if x is None: raise RuntimeError(...)` constructs. Improves `python -O` compatibility and diagnostic clarity. File a follow-up chore commit or defer to a future hardening PR.

2. **Finding #6 (backlog, effort S):** Add a `SECURITY.md` note: "Load model artifacts only from trusted sources. Chat-template injection via maliciously-constructed tokenizer Jinja2 templates is possible but runs inside an `ImmutableSandboxedEnvironment`, limiting impact to prompt manipulation (not arbitrary code execution)."

3. **Finding #5 (backlog, effort XS):** Add a one-line docstring to each `@runtime_checkable` Protocol: "Structural existence checks only; do not use as a trust gate for untrusted input."

**Process concern (NOT a security finding, surfaced during audit):**
- `.dialectic-tier` is tracked in `git` on this feature branch, while `.gitignore` only excludes the legacy name `.push-hands-tier`. Per CLAUDE.md: tier metadata must never reach `main`. Addressed in the audit commit alongside this report.

## Summary

Execute phase delivered a clean mypy-strict pass with **zero security regressions** and two net-positive improvements: the cross-framework conversion removal (CLAUDE.md invariant now architecturally enforced) and the `bridge.py` latent-bug fix (real attribute access + regression test, replacing MagicMock-masked `AttributeError`). The dialectic properly downgraded two LOW findings to INFO where the auditor's own prose acknowledged no diff-induced exposure. The single surviving LOW is optional `-O`-mode hardening.

**Verdict: APPROVED for merge from a security standpoint.**
