# PRP: Security Audit Remediation — tgirl.compile

## Context

The security audit (`docs/audits/compile-audit.md`) identified 8 findings (post-challenge: 0 HIGH, 2 MEDIUM, 4 LOW, 2 INFO). Additionally, the user identified a critical gap: the Hy AST analyzer does not reject all compile-time metaprogramming forms (`defmacro/g!`, `eval-and-compile`, `eval-when-compile`, `include`), which could execute code during `hy.compile()` before the Python AST analyzer or sandbox ever sees it. This is the **macro-expansion trap** — Hy macros execute at compile time, not runtime.

This plan addresses the 6 actionable findings (priorities 1-6) plus the metaprogramming firewall gap.

## Files to modify

- `src/tgirl/compile.py` — all fixes
- `tests/test_compile.py` — new tests for each fix

## Test Command

```bash
pytest tests/test_compile.py tests/test_integration_compile.py -v
```

## Verification

```bash
pytest tests/test_compile.py tests/test_integration_compile.py -v
ruff check src/tgirl/compile.py tests/test_compile.py
```

All 244+ existing tests must continue to pass. New tests added for each fix.

## Implementation Tasks

### Task 1: Harden Hy AST metaprogramming firewall (user-identified, CRITICAL defense-in-depth)

**File:** `src/tgirl/compile.py:198-203`

**Problem:** `_DEFINITION_FORMS` and `_IMPORT_FORMS` are incomplete. Missing: `defmacro/g!`, `eval-and-compile`, `eval-when-compile`, `include`. These forms execute code during `hy.compile()` — before the Python AST analyzer or sandbox. This is the macro-expansion trap.

**Fix:**
- Merge `_DEFINITION_FORMS` and `_IMPORT_FORMS` into a single `_DISALLOWED_FORMS` frozenset containing ALL dangerous head symbols:
  ```
  defn, defmacro, defmacro/g!, defclass, deftype,
  import, require, include,
  eval-and-compile, eval-when-compile
  ```
- Update `_check_node` (line 229-244) to use the unified set
- Also reject any Expression whose head is a Symbol starting with `.` (method call syntax `(.method obj)`) — this feeds into Finding 4

**Tests:**
- `defmacro/g!` form rejected
- `eval-and-compile` form rejected
- `eval-when-compile` form rejected
- `include` form rejected
- `.method` syntax rejected at Hy AST level

### Task 2: Close method call bypass in `_check_node` (Finding 4, MEDIUM)

**File:** `src/tgirl/compile.py:225, 380-384`

**Problem:** When the head of an Expression is itself an Expression (not a Symbol), `_check_node` skips call target validation entirely (line 225 only matches Symbol). This allows `(.upper "hello")` and critically `(.format "{0.__globals__}" tool_fn)` to pass.

**Fix:**
- After the `isinstance(head, Symbol)` block (line 225-378), add an `elif isinstance(head, Expression)` block that rejects Expression-headed calls with error type `"DisallowedForm"` and message `"Method call syntax is not allowed"`.
- This is belt-and-suspenders with Task 1's `.`-prefix check — Task 1 catches `(.method ...)` at the symbol level, Task 2 catches any remaining Expression-headed calls structurally.

**Tests:**
- `(.upper "hello")` rejected
- `(.format "{0.__globals__}" f)` rejected
- Nested method call in composition rejected
- Legitimate tool calls still pass

### Task 3: Fix timeout enforcement (Finding 2, MEDIUM)

**File:** `src/tgirl/compile.py:533-554, 557-580`

**Problem:** `ThreadPoolExecutor`'s `with` statement calls `shutdown(wait=True)` on exit, blocking until the submitted task completes — defeating the timeout. PoC: 0.2s timeout takes 5.0s.

**Fix:**
- In `_run_with_timeout`: replace `with` statement with manual executor management. Call `executor.shutdown(wait=False, cancel_futures=True)` after timeout.
  ```python
  executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
  future = executor.submit(fn)
  try:
      return future.result(timeout=timeout)
  except concurrent.futures.TimeoutError:
      return PipelineError(...)
  finally:
      executor.shutdown(wait=False, cancel_futures=True)
  ```
- Same pattern in `_wrap_with_timeout`.
- Note: background threads still run (Python limitation) but the caller is no longer blocked.

**Tests:**
- Slow tool (5s sleep) with 0.5s timeout returns PipelineError in < 1s (not 5s)
- Per-tool timeout wrapper also returns within timeout bound
- Fast execution still completes normally

### Task 4: Fix `bound_vars` scope leakage (Finding 1, LOW)

**File:** `src/tgirl/compile.py:217, 417-420`

**Problem:** `bound_vars` is a single `set[str]` shared across all top-level trees. Variables bound in one tree leak into subsequent trees.

**Fix:**
- Initialize fresh `bound_vars = set()` inside the `for tree in trees` loop (line 417), not outside it.
- Alternatively, pass `bound_vars` as a parameter to `_check_node` and create fresh copies at scope boundaries (let blocks).

**Tests:**
- Variable bound in first tree is not visible in second tree
- Variable bound in let is still visible within same let block
- Let-bound variable not visible outside let block (if implementing scope stack)

### Task 5: Add List recursion to `_expand_macros` (Finding 3, LOW)

**File:** `src/tgirl/compile.py:146-183`

**Problem:** `_expand_macros` only recurses into Expression children, not List nodes. Threading (`->`) inside a pmap list won't be expanded.

**Fix:**
- Add a `List` case alongside the existing Expression recursion (around line 177-183):
  ```python
  elif isinstance(node, List):
      return List([_expand_macros(child) for child in node])
  ```

**Tests:**
- `(pmap [(-> x (tool1) (tool2)) tool3] arg)` — threading inside pmap list is expanded correctly

### Task 6: Catch `SystemExit` in sandbox execution (Finding 7, LOW)

**File:** `src/tgirl/compile.py:778-788`

**Problem:** `_execute` catches `Exception` but not `SystemExit`. A tool raising `SystemExit` kills the host process.

**Fix:**
- Change `except Exception as exc:` to `except (Exception, SystemExit) as exc:`
- Do NOT catch `KeyboardInterrupt` (must preserve Ctrl+C) or `GeneratorExit`

**Tests:**
- Tool that raises `SystemExit(0)` returns PipelineError (not process exit)
- `KeyboardInterrupt` still propagates (negative test — verify it's NOT caught)

### Task 7: Remove dead `max_depth` config (Finding 5, LOW)

**File:** `src/tgirl/compile.py:68`

**Problem:** `CompileConfig.max_depth = 50` is defined but never referenced.

**Fix:**
- Remove the `max_depth` field from `CompileConfig`

**Tests:**
- Update `TestCompileTypes` to verify `CompileConfig` has only `pipeline_timeout`
