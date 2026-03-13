# Security Audit: compile

## Scope
**Examined:** `src/tgirl/compile.py` (812 lines), `tests/test_compile.py` (751 lines), `tests/test_integration_compile.py` (249 lines), `src/tgirl/__init__.py` changes. All 14 commits on `feature/compile` branch.

**NOT examined:** `grammar.py`, `transport.py`, `sample.py`, `bridge.py`, `serve.py` (not in diff). Dependency versions/CVEs. Race conditions in concurrent pipeline execution. Memory exhaustion via large string literals (grammar should prevent).

## Methodology
Dual-agent team audit — Security Auditor and Skeptical Client operating as separate agents with direct peer messaging. Findings were challenged in real time, producing severity ratings that survived adversarial scrutiny. The auditor provided proof-of-concept scripts for both HIGH findings. The client evaluated each finding against the three-layer defense model (grammar + static analysis + sandbox).

Six items flagged by the code review were explicitly audited, plus independent vulnerability hunting across input validation, data exposure, configuration, business logic, and DoS categories.

## Findings Summary

| # | Initial | Final | Category | Description | Effort |
|---|---------|-------|----------|-------------|--------|
| 4 | HIGH | **MEDIUM** | Data Exposure | Method call syntax bypasses call target validation; `str.format` leaks `__globals__` | M |
| 2 | HIGH | **MEDIUM** | DoS | Timeout mechanisms don't actually bound execution time | M |
| 1 | MEDIUM | **LOW** | Business Logic | `bound_vars` flat scoping leaks variables across expressions | S |
| 7 | MEDIUM | **LOW** | DoS | `BaseException` propagation kills host process | XS |
| 3 | LOW | **LOW** | Business Logic | `_expand_macros` doesn't recurse into List nodes | XS |
| 5 | LOW | **LOW** | Config | `max_depth` config defined but never enforced | S |
| 6 | INFO | **INFO** | Config | No reserved name validation on tool registration | XS |
| 8 | INFO | **INFO** | Data Exposure | Sensitive data (tool arguments) in pipeline logs | S |

**Post-challenge: 0 CRITICAL, 0 HIGH, 2 MEDIUM, 4 LOW, 2 INFO**

## Detailed Findings

### Finding 1: `bound_vars` flat scoping leaks variables across expressions
**Initial Severity:** MEDIUM
**Final Severity:** LOW
**Category:** Business Logic
**Affected Code:** `src/tgirl/compile.py:217, 277-283, 314`
**Description:** `bound_vars` is a single flat `set[str]` shared across all `_check_node` calls. Variables bound in one `let` block leak into sibling expressions, making the static analyzer more permissive than intended.
**Proof of Concept:** Parse `(let [x (greet "a")] x)` and `(shout x)` separately — `x` fails. Combine them — `x` passes because it was added by the first expression.
**Client Challenge:** Grammar `start` rule produces exactly one expression — no repetition. Multi-expression input can't come from grammar-constrained output. Even if it did, Hy runtime scoping catches it with `NameError`. Two layers independently prevent exploitation.
**Auditor Defense:** Acknowledged grammar prevents multi-expression output. Maintained as defense-in-depth concern.
**Resolution:** Downgraded to LOW. Grammar and runtime scoping both prevent exploitation. Static analyzer permissiveness is a correctness issue, not a security vulnerability.
**Remediation:** Use a scope stack or analyze each top-level tree with fresh `bound_vars`.
**Effort Estimate:** S

### Finding 2: Timeout mechanisms don't actually bound execution time
**Initial Severity:** HIGH
**Final Severity:** MEDIUM
**Category:** DoS / Configuration
**Affected Code:** `src/tgirl/compile.py:533-554` (`_run_with_timeout`), `557-580` (`_wrap_with_timeout`)
**Description:** `ThreadPoolExecutor` used with `with` statement calls `shutdown(wait=True)` on context manager exit. The caller blocks for the full tool execution time despite `future.result(timeout=...)` raising `TimeoutError`.
**Proof of Concept:** `CompileConfig(pipeline_timeout=0.2)` with a tool sleeping 5 seconds — pipeline takes 5.0 seconds, not 0.2 seconds.
**Client Challenge:** Tools are trusted developer code, not model-controlled — "malicious tool" is the wrong threat model. `serve.py` doesn't exist yet, so the DoS scenario is speculative. The fix (`shutdown(wait=False)`) is inherently incomplete since Python threads can't be killed. True timeout enforcement belongs in `serve.py`'s process isolation design.
**Auditor Defense:** The PoC is unambiguous — timeouts don't work as advertised. Even if tools are trusted, buggy tools with network calls or database queries can stall indefinitely. The timeout exists for a reason and should function correctly.
**Resolution:** Downgraded to MEDIUM. Valid correctness bug with demonstrated PoC. Not a security vulnerability in current scope (no server, trusted tools), but should be fixed because the feature should work as documented.
**Remediation:** Use `shutdown(wait=False, cancel_futures=True)`. Design proper process isolation when `serve.py` is implemented.
**Effort Estimate:** M

### Finding 3: `_expand_macros` doesn't recurse into List nodes
**Initial Severity:** LOW
**Final Severity:** LOW
**Category:** Business Logic
**Affected Code:** `src/tgirl/compile.py:146-183`
**Description:** `->` (thread-first) inside a `List` node (e.g., inside `pmap`'s function list) is not expanded.
**Client Challenge:** Accepted at LOW. Noted that grammar's `pmap_expr` uses `expr` not `SYMBOL` inside brackets, so grammar could produce `->` inside pmap. Fails safely at runtime (TypeError, not security breach). Also identified related gap: `_analyze_hy_ast` silently skips non-Symbol entries in pmap's function list.
**Resolution:** Maintained at LOW. Correctness bug, not security.
**Remediation:** Add `List` recursion case to `_expand_macros`. Consider tightening `pmap_expr` grammar to `SYMBOL`.
**Effort Estimate:** XS

### Finding 4: Method call syntax bypasses call target validation; `str.format` leaks `__globals__`
**Initial Severity:** HIGH
**Final Severity:** MEDIUM
**Category:** Data Exposure
**Affected Code:** `src/tgirl/compile.py:219-384` (`_check_node`), `481-495` (`visit_Attribute`)
**Description:** Hy method call syntax `(.method obj)` parses with an Expression head, not a Symbol head. `_check_node` only validates call targets when `isinstance(head, Symbol)`. Method calls bypass tool registration checks entirely. Combined with `str.format`, format specs like `{0.__class__}` traverse dunder attributes without triggering `visit_Attribute`, since attribute access happens inside CPython's `str.format` implementation, not at the Python AST level.
**Proof of Concept:**
```python
run_pipeline('(.upper "hello")', reg)  # => "HELLO" — arbitrary method calls
run_pipeline('(let [f echo] (.format "{0.__globals__}" f))', reg)
# => dumps tool's entire defining module scope
```
**Client Challenge:** Grammar has NO production for `.method` syntax — checked every template. Model cannot generate `(.format ...)` under grammar constraint. PoC requires hand-crafted input bypassing the grammar entirely, which is not the threat model. However, the `_check_node` gap for Expression-headed calls is a real defense-in-depth weakness, and `str.format` dunder traversal is a known technique that matters if attribute access productions are ever added.
**Auditor Defense:** Grammar is the primary guard, but defense-in-depth is the design philosophy. If grammar is ever loosened (e.g., method calls on typed return values), this becomes exploitable. The PoC demonstrates the static analyzer doesn't catch it.
**Resolution:** Downgraded to MEDIUM. Grammar prevents exploitation in current architecture. Defense-in-depth gap is real and should be closed — especially given the `str.format` dunder traversal technique which bypasses AST-level attribute checking entirely.
**Remediation:** (1) Detect Expression heads in `_check_node` and reject or validate method names against an allowlist. (2) Defense-in-depth: wrap tool callables before placing in sandbox to isolate `__globals__`.
**Effort Estimate:** M

### Finding 5: `max_depth` config defined but never enforced
**Initial Severity:** LOW
**Final Severity:** LOW
**Category:** Configuration
**Affected Code:** `src/tgirl/compile.py:68`
**Description:** `CompileConfig.max_depth = 50` exists but is never referenced in the pipeline. Depth-200 nesting takes ~2.8s; depth-500 hits Python's recursion limit (caught as parse error).
**Client Challenge:** Accepted at LOW. Dead config field. Grammar constrains nesting, Python recursion limit provides backstop.
**Resolution:** Maintained at LOW.
**Remediation:** Enforce during Hy AST analysis, or remove the field with a TODO.
**Effort Estimate:** S

### Finding 6: No reserved name validation on tool registration
**Initial Severity:** INFO
**Final Severity:** INFO
**Category:** Configuration
**Affected Code:** `src/tgirl/compile.py:583-623`, `src/tgirl/registry.py:66-101`
**Description:** Tools can be registered with names colliding with sandbox internals (`pmap`, `insufficient_resources`, `_tgirl_result_`, `__builtins__`). The sandbox silently overwrites them.
**Client Challenge:** Accepted at INFO. Usability issue. Fix belongs in `registry.py`, not `compile.py`.
**Resolution:** Maintained at INFO.
**Remediation:** Add reserved-names check in `ToolRegistry.tool()`.
**Effort Estimate:** XS

### Finding 7: `BaseException` propagation kills host process
**Initial Severity:** MEDIUM
**Final Severity:** LOW
**Category:** DoS
**Affected Code:** `src/tgirl/compile.py:778-788`
**Description:** `_execute` catches `Exception` but `SystemExit` and `KeyboardInterrupt` (inheriting `BaseException`) propagate uncaught.
**Proof of Concept:** A tool raising `SystemExit(0)` propagates out of `run_pipeline` and kills the host process.
**Client Challenge:** Model cannot raise `SystemExit` — it's not in sandbox builtins, and all three layers block `raise` statements. Requires a buggy tool (trusted developer code). `KeyboardInterrupt` should NOT be caught — that would break Ctrl+C.
**Auditor Defense:** Acknowledged — `SystemExit` is not constructable from Hy. Changed recommendation to catch `SystemExit` specifically, not `BaseException`.
**Resolution:** Downgraded to LOW. Requires buggy tool code, not model-controllable.
**Remediation:** Change to `except (Exception, SystemExit)` specifically. Do NOT use `except BaseException`.
**Effort Estimate:** XS

### Finding 8: Sensitive data in pipeline logs
**Initial Severity:** INFO
**Final Severity:** INFO
**Category:** Data Exposure
**Affected Code:** `src/tgirl/compile.py:101, 705, 803-807`
**Description:** Full Hy source (including string literal arguments) logged at WARNING, DEBUG, and INFO levels. Tool arguments may contain PII or secrets.
**Client Challenge:** Accepted at INFO. Standard log hygiene.
**Resolution:** Maintained at INFO.
**Remediation:** Redact arguments or reserve full source for DEBUG only.
**Effort Estimate:** S

## Additional Notes from Audit

- **Chained attribute access:** Hy AST only checks `node[2]` for dunders in `(. obj attr1 attr2)` form — `node[3+]` dunders are not checked. Python AST backstop (`visit_Attribute`) catches these. Defense-in-depth gap noted but covered by layer 2.
- **`_find_hy_anon_var` correctness:** Uses `ast.walk` returning first match. Hy chains anon vars (`_hy_anon_var_2 = _hy_anon_var_1`), so first match is always valid. Not a security issue.
- **`_normalize_hy_source` regex:** Correctly uses `\b` word boundary. `(catcher ...)` is not affected.
- **Safe builtins DoS surface:** `range`, `list`, `dict` are in `__builtins__` but cannot be called as Hy function call targets (blocked by Hy AST as unregistered tools). DoS via these builtins requires grammar loosening.

## What This Audit Did NOT Find
- No authentication/authorization issues (module has no auth surface)
- No cryptographic weaknesses (module has no crypto)
- No injection vulnerabilities beyond the documented defense-in-depth gaps
- No exploitable sandbox escapes — all bypass vectors require grammar-layer failure

## Remediation Priority

1. **Finding 4** (MEDIUM, M) — Close method call bypass in `_check_node`. Largest defense-in-depth gap.
2. **Finding 2** (MEDIUM, M) — Fix timeout to use `shutdown(wait=False, cancel_futures=True)`. Correctness.
3. **Finding 1** (LOW, S) — Fresh `bound_vars` per top-level tree. Static analyzer correctness.
4. **Finding 3** (LOW, XS) — Add List recursion to `_expand_macros`. Correctness.
5. **Finding 7** (LOW, XS) — Catch `SystemExit` specifically. Quick fix.
6. **Finding 5** (LOW, S) — Remove `max_depth` or enforce it. Dead code cleanup.
7. **Finding 6** (INFO, XS) — Reserved names check in registry. Deferred.
8. **Finding 8** (INFO, S) — Log level hygiene. Deferred.

**Total estimated effort for priorities 1-6:** 2M + 2S + 2XS
