# Code Review: mypy-strict-cleanup

## Verdict: APPROVED
## Reviewer Stance: Team — Proposer + Code Review Partner
## Date: 2026-04-23
## Mode: Agent Team (message-gated incremental review)

## Summary

All 14 PRP tasks implemented, each as an atomic commit, each individually reviewed. Zero Blocking findings, zero Significant findings required rework. `mypy src/` went from 100 errors to 0; full test suite stayed green at 1123 passed (5 new tests added by the PRP's TDD requirement). `# type: ignore` budget consumed 4 of 10 allocated comments. Two substantive latent bugs were caught during the type-tightening pass and fixed with regression tests.

## PRP Compliance

| # | Task | SHA | mypy Δ | Implemented as spec'd? |
|---|---|---|---|---|
| 1 | Baseline snapshot + third-party mypy overrides | `4c07b49` | 100→100 | Yes |
| 2 | Add missing generic type parameters (type-arg cluster) | `de24307` | 100→86 | Yes — overshoot explained (one no-any-return dissolved as side-effect) |
| 3 | Delete stale `# type: ignore` comments | `f49eba9` | 86→81 | Yes |
| 4 | Add missing return/argument annotations | `18e172b` | 81→74 | Yes |
| 5 | `EstradiolControllerProto` + `ConfidenceMonitorProto` | `b192470` | 74→63 | Yes — both Protocols co-located with concrete classes |
| 6 | Steerable forward-fn typing | `9112849` | 63→57 | **Deviated (documented) —** cast-based narrowing instead of factory unification. PRP §6 escape clause explicitly authorized (PRP threshold: "split if >3 call sites ripple"; actual ripple: 40 test failures / 29 mock sites = 10× threshold). Latent caller contract preserved as runtime TypeError, not static guarantee. |
| 7 | `TokenizerProto` for format + outlines_adapter | `0477cda` | 57→49 | Mostly — `format.py` got the Protocol. `outlines_adapter.py` used `tokenizer: Any` instead of partial Protocol (rationale: avoid re-declaring llguidance's `TokenizersBackend` surface). Documented in commit body. |
| 8 | Narrow `InferenceHook` attribute access | `f177f6c` | 49→47 | Yes — `isinstance(hook, ModMatrixHook)` narrowing |
| 9 | `no-any-return` cluster | `fe4193c` | 47→39 | Yes **+ latent bug found and fixed** (see Issue #2 below) |
| 10 | Union narrowing & scope-shadowing cleanup | `884468d` | 39→20 | Yes. Chose `cast(Any, mx.cpu)` over `# type: ignore[arg-type]` for the 4 MLX stream-arg sites — preserves ignore budget at slight loss of type precision |
| 11 | Narrow `kwargs.get('token_id')` before `int()` | `3c5d416` | 20→18 | Yes |
| 12 | Latent-bug cluster | `f73e8ce` | 18→15 | **Exemplary.** Confirmed PRP's possibility 2 for rerank: `ToolRouter` only held the torch factory. Fix is architectural — added `mlx_grammar_guide_factory` parameter, wired through `SamplingSession`, dispatched in `ToolRouter.route()`. CLAUDE.md-violating cross-framework conversion (2026-03-14 gotcha) DELETED, replaced with fail-loud `TypeError` + remediation message. Regression test locks the contract. (See Issue #3.) |
| 13 | Remaining arg-type / call-arg / misc | `2b3dc00` | 15→0 | Yes. mypy reaches zero. |
| 14 | Cleanup — delete baseline snapshot | `f5f8b84` | 0→0 | Yes |

**Terminal gates met:** `mypy src/` → `Success: no issues found in 33 source files`. `pytest tests/` → 1123 passed. `# type: ignore[rule]` budget: **4 of 10** used (outlines_adapter.py:145,149 for llguidance.mlx stubs; serve.py:91 for mlx_lm.load stub; compile.py:439 for RestrictedPython dynamic base).

## Issues Found

### 1. Task 6 PRP deviation — factory unification deferred
**Category:** Architecture
**Severity:** Significant (accepted)
**Location:** `src/tgirl/sample_mlx.py` (commit `9112849`)
**Details:** PRP Task 6 originally prescribed unifying `make_mlx_forward_fn` and `make_steerable_mlx_forward_fn` so both return `ForwardResult`. Empirically, that unification rippled to 40 test failures across 29 inline mock `forward_fn` sites in test_sample, test_sample_mlx, test_state_machine, test_rerank, test_integration_sample, test_sample_rerank — 10× the PRP's stated escape threshold. Proposer invoked PRP §6 escape clause ("if updating the factory signature breaks more than 3 call sites, split Task 6") and used `cast(SteerableForwardFn, forward_fn)` at the steered branch instead. Same 4 errors resolved, smaller blast radius.
**Resolution:** Accepted as documented deviation. The latent caller contract (controller must be paired with a `SteerableForwardFn`) becomes a runtime TypeError rather than a compile-time type guarantee — but that constraint pre-existed this change and isn't a regression. **Follow-up tracked in Uncertainty Log:** factory unification is appropriate for a future dedicated PR once test infrastructure has a standard `ForwardResult`-aware mock factory.

### 2. `bridge.py:538` latent bug — `.text` vs `.output_text`
**Category:** Logic (latent bug)
**Severity:** Blocking at runtime, Silent in tests (discovered during mypy cleanup)
**Location:** `src/tgirl/bridge.py:538` (commit `fe4193c`)
**Details:** The FastMCP handler referenced `result.text`, but `SamplingResult` exposes `.output_text`. This was a real `AttributeError` in production that had been silently masked in tests by `MagicMock`'s auto-attribute synthesis — the mock returned a Mock object, the handler called `.strip()` on it, and the test passed with a non-deterministic stringified Mock. `no-any-return` narrowing in Task 9 forced the proposer to declare the concrete return type, which surfaced the mismatch.
**Resolution:** Fixed in the same commit. Existing mock test updated to use a real `SamplingResult`. New regression test `test_expose_as_mcp_uses_output_text_field_on_real_sampling_result` constructs a real `SamplingResult` through FastMCP's `call_tool` end-to-end, locking the field contract.

### 3. `sample_mlx.py:506-507` CLAUDE.md invariant violation — cross-framework conversion
**Category:** Architecture / Convention (CLAUDE.md 2026-03-14 gotcha)
**Severity:** Blocking at the invariant level; silent at runtime (only fired in the untested MLX+torch rerank path)
**Location:** `src/tgirl/sample_mlx.py:506-507` → fixed via `src/tgirl/rerank.py`, `src/tgirl/sample.py` (commit `f73e8ce`)
**Details:** Fallback code path did `valid_mask = mx.array(valid_mask_torch.numpy())` — exactly the torch→numpy→mx conversion CLAUDE.md explicitly forbids. The fallback only fires in a rerank path that wasn't covered by tests. Had gone unnoticed because `ToolRouter` only held a torch grammar-guide factory; the MLX path fell through to this conversion silently.
**Resolution:** Architectural fix. `ToolRouter` now takes a `mlx_grammar_guide_factory` parameter; `SamplingSession` threads both factories; `ToolRouter.route()` dispatches on backend. The cross-framework conversion fallback **deleted** in favor of a fail-loud `TypeError` with a remediation message. Regression test `test_mlx_backend_uses_mlx_grammar_state` locks three invariants (MLX factory called, torch factory not called, correct grammar_state threaded through). Old `test_torch_grammar_fallback` inverted to `test_torch_grammar_state_raises_typeerror`, validating the fail-loud contract.

### 4. `outlines_adapter.py` tokenizer typing less precise than PRP prescribed
**Category:** Type precision / Documentation
**Severity:** Minor
**Location:** `src/tgirl/outlines_adapter.py:105, 181` (commit `0477cda`)
**Details:** PRP prescribed a partial `TokenizerProto` surfacing `vocab`, `vocab_size`, `encode`/`decode`. Proposer used `tokenizer: Any` instead, rationale: llguidance's downstream `from_tokenizer` already enforces `TokenizersBackend`, and re-declaring a partial surface here would duplicate an imperfect subset of that.
**Resolution:** Accepted. Trade-off is documentation-only: the Protocol form would have made call-site expectations locally legible. Not worth revisiting.

### 5. `PromptFormatter.format_messages` signature widened to `**kwargs: object`
**Category:** API contract
**Severity:** Minor
**Location:** `src/tgirl/types.py` (commit `2b3dc00`)
**Details:** The Protocol was widened to accept arbitrary kwargs as part of resolving a call-arg error. This is an API surface change — any external implementer of `PromptFormatter` is now implicitly required to accept kwargs too.
**Resolution:** Accepted. The widening matches what the concrete `ChatTemplateFormatter` already does (it passes `**kwargs` through to `tokenizer.apply_chat_template`). Worth calling out in release notes.

### 6. Behavior shift on non-int/str `token_id` in `state_machine.py`
**Category:** Logic
**Severity:** Minor
**Location:** `src/tgirl/state_machine.py` (commit `3c5d416`)
**Details:** Previously, `int(kwargs.get("token_id"))` on a non-int/str would raise `TypeError`. The new `isinstance` narrowing silently returns a no-transition `TransitionDecision` instead.
**Resolution:** Accepted — the new behavior is strictly more defensive (fewer crashes on pathological caller inputs), which aligns with a session's "don't crash mid-generation" disposition. Documented in commit message.

### 7. Minor commit message attribution slip
**Category:** Documentation
**Severity:** Nit
**Location:** commit `884468d` body
**Details:** The commit body attributed an unused-import cleanup to Task 10; it was actually in Task 9.
**Resolution:** Not worth amending. Git trailers are accurate; body prose is informational.

## What's Done Well

- **TDD discipline.** Every task shipped with a RED signal (mypy error count for the subset) and a GREEN signal (count → 0 or target delta). 5 new tests added across the change, all locking behavior (protocol satisfaction, regression for latent bugs, invariants for the cross-framework fix).
- **Deviation transparency.** The Task 6 deviation is documented in the commit message with specific evidence (40 test failures, 29 mock sites), cites the exact PRP clause that authorized the escape, and quantifies the cost (latent contract preserved as runtime TypeError).
- **Gold-standard architectural fix in Task 12.** The CLAUDE.md cross-framework-conversion violation wasn't just removed — it was replaced with an architectural solution (backend-aware factory in `ToolRouter`) plus a fail-loud contract enforced by test inversion. This is how you remove a shim.
- **Budget discipline.** `# type: ignore` budget: 4 of 10. Every ignore has a `[rule]` specifier and a one-line upstream-gap reason. No bare `# type: ignore`.
- **Monotonic mypy decrease.** Commit footers track `mypy: X → Y (delta -N)`. Every commit's delta was verified by the reviewer; no regressions through the chain.
- **Latent-bug discovery.** Task 9's `no-any-return` cleanup accidentally caught a real production bug (`result.text` → `result.output_text`). Proposer did not treat it as out-of-scope; they fixed it, updated the mock test, and added a regression test exercising the real `SamplingResult` path end-to-end. Exactly the "type-correctness work catches real bugs" scenario the PRD argued for.
- **Per-task mypy gate enforcement.** The "interlocutor runs mypy between commits" gate from the plan review landed; reviewer verified every commit's mypy delta in LGTM messages.

## Summary

The execute phase converged cleanly on its terminal goal. Zero Blocking findings, one Significant-but-accepted PRP deviation (with an explicit escape-clause citation), a handful of Minor trade-offs (accepted), and two latent production bugs caught and fixed with regression tests. The team followed the message-gated cadence faithfully — the proposer respected findings even when reviewer responses arrived slightly outside the 2-minute window, and the reviewer produced a substantive per-commit audit with verified mypy deltas.

**Verdict: APPROVED.** Recommend direct PR creation or optional `/review-code-team` if a fresh pair of eyes would be useful on the Task 6 deviation and the `PromptFormatter` API widening.
