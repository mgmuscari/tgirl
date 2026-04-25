# Security Audit: plugin-architecture

## Verdict: REJECT for merge

Three independent CRITICAL sandbox-escape paths confirmed at zero declared capability. The two-key opt-in is decorative in the current implementation state. Capability-isolation claims in the PRD/PRP are not satisfied.

The PRP design is sound; the implementation is incomplete. Findings #4 and #5 are PRP-specified-but-unimplemented (the Sandbox B `__builtins__` substitution and `getattr` AST ban). Do NOT reopen the PRP — fix the implementation drift and re-audit.

## Scope

**Examined:**
- 10 commits on `feature/plugin-architecture` (Tasks 1, 2, 3, 4, 10, 5, 6, 7 + 2 cleanup commits).
- All capability-cluster source: `src/tgirl/plugins/{__init__,types,config,loader,ast_scan,guard,capability_modules,errors}.py`, `src/tgirl/plugins/capabilities/{env_proxy,subprocess_proxy,fs_read_proxy,fs_write_proxy}.py`, `src/tgirl/compile.py` capability-relaxation knob.
- Sandbox B contract per PRP §Task 4 (safe-builtins substitution, AST scan FORBIDDEN_NAMES, three-gate model).
- Capability-to-module mapping completeness, proxy module exclusions, AST `_check_getattr_call` narrowing.
- TOML config parser path-handling.
- Lark grammar generation tool-name handling.
- Adversarial cases: dynamic import, dunder-string construction, type-mro reflection, `getattr` indirection.

**NOT examined (explicit limitations):**
- MLX/Torch sample/transport modules — no plugin coupling, out of scope.
- `bridge.py` (MCP layer) — unchanged on this branch.
- `compile.py`'s `_TgirlNodeTransformer` for LLM-generated Hy code (Sandbox A) — only the new `grant` plumbing was reviewed; full Hy/Python AST scan from prior PRs not re-audited.
- Exhaustive CPython 3.11/3.12/3.13 reflection paths (`__init_subclass__`, `__class_getitem__`, descriptor protocol via `type.__call__`) — Sandbox B's blanket dunder-attribute reject mostly closes this but full enumeration deferred.
- `serve.py` request flow — Task 11 not shipped on this branch.
- Telemetry endpoint privacy gate (Task 11 §Y8) — not shipped.
- Stdlib version pinning (Task 11 §Y7) — not shipped.
- Hypothesis property test runner — verified file exists; full CI profile not exercised.
- TOML billion-laughs / deep nesting / memory exhaustion stress test — basic robustness verified.
- Locale / tokenizer edge cases for `_sanitize_rule_name` end-to-end through `llguidance`.
- Subprocess-proxy `shell=True` audit — per-PRP-spec full RCE on SUBPROCESS grant; documented behavior, not flagged.

## Methodology

Dual-agent team audit — Security Auditor (hard stance, exploit-oriented) and Skeptical Client (hard stance, severity-calibrating) operating as separate agents with direct peer messaging. Each finding followed a fixed protocol: auditor reports with PoC → client challenges → auditor defends or acknowledges → finding closes at agreed severity. No oscillation, one round per finding. HIGH+ findings required concrete proof-of-concept demonstrating exploitation, not just descriptive analysis.

## Findings Summary

| # | Final Severity | Category | Description | Effort |
|---|---|---|---|---|
| 1 | MEDIUM | Authorization (runtime correctness) | Phase B unimplemented; granted plugins fail-closed at first gated call | S |
| 2 | CRITICAL | Sandbox / privilege escalation | Zero-cap plugin mutates `CAPABILITY_MODULES` dict to remap modules to default-granted capability | S |
| 3 | MEDIUM | Concurrency / race conditions | `guard_scope` mutates global `sys.modules` (CRITICAL once Task 11 lands) | S |
| 4 | CRITICAL | Sandbox / privilege escalation | Sandbox B `__builtins__` never restricted; raw `open()` available at zero capability | S |
| 5 | CRITICAL | Sandbox / privilege escalation | `getattr` + `__globals__` reflection chain bypasses all gates at zero capability | S |
| 6 | MEDIUM | Input validation | Lark grammar terminal injection via unescaped tool name | XS |
| 7 | MEDIUM | Input validation / path traversal | TOML `module` field accepts path traversal | XS |
| 8 | MEDIUM | Sandbox / spec drift | `breakpoint`/`input` missing from FORBIDDEN_NAMES (PRP §2b spec drift) | XS |
| 9 | LOW | Code quality | `_register_tools` bypasses `ToolRegistry` public API | S |
| 10 | LOW | Functional bug | `env_proxy.__contains__` broken through `CapabilityScopedModule` wrapper | XS |
| 11 | INFO | Documentation | `fs_write_proxy.rename` exposed per-spec; type annotation drift | XS |
| 12 | MEDIUM | Test coverage | Security test cluster never invokes registered tools (causally responsible for hiding #1) | S |

**Aggregate:** 3 CRITICAL, 0 HIGH, 6 MEDIUM, 2 LOW, 1 INFO.

## Detailed Findings

### Finding 1: Phase B unimplemented; granted plugins cannot use granted capabilities
**Initial Severity:** CRITICAL
**Final Severity:** MEDIUM (security) + spawned #12 (MEDIUM, dev-readiness)
**Category:** Authorization (runtime correctness)
**Affected Code:** `src/tgirl/plugins/loader.py:260` (sole `guard_scope` call site); `src/tgirl/plugins/registry.py:395` (`get_callable` returns raw); `src/tgirl/plugins/loader.py:184-204` (`_register_tools` writes raw callable).
**Description:** `_effective_grant` contextvar is never set during runtime tool dispatch. Every callable attribute on a `CapabilityScopedModule` reads `default=None` → empty granted capabilities → all callables denied.
**Proof of Concept:** Granted plugin with `allow={NETWORK}` and runtime grant `{NETWORK,CLOCK,RANDOM}` calls `socket.socket()` from a tool function → `CapabilityDeniedError: capability 'network' required for 'socket.socket'`. Verified empirically by both proposer and auditor.
**Client Challenge:** Severity. Bug is real but: (a) Task 9 is already scoped to fix this per PRP, (b) it fails CLOSED — granted plugins cannot exploit anything because they cannot run gated functions at all. Re-classified as a "the system doesn't work" bug, not a "the system has a security hole" bug. Spawned new #12 to capture the test-coverage gap that hid this.
**Auditor Defense:** Acknowledged the runtime-fail-closed framing.
**Resolution:** Downgraded to MEDIUM (security) + spawned #12 (MEDIUM, dev-readiness). Phase B wiring still required before plugins are usable but is not an active exploitation vector.
**Remediation:** Wrap callables in `_register_tools` with `guard_scope(grant)`. Better: refactor to remove `sys.modules` mutation entirely, using thread-local active-grant lookup in the meta_path finder. Latter approach also resolves #3.
**Effort Estimate:** S.

### Finding 2: Zero-cap plugin escalates to arbitrary capability via dict mutation
**Initial Severity:** CRITICAL
**Final Severity:** CRITICAL
**Category:** Sandbox / privilege escalation
**Affected Code:** `src/tgirl/plugins/capability_modules.py:34` (mutable `CAPABILITY_MODULES: dict`); `src/tgirl/plugins/capability_modules.py:80` (`ALWAYS_ALLOWED_MODULES` matches root `tgirl`).
**Description:** `tgirl.plugins.capability_modules` is reachable at zero grant because the wildcard `tgirl` root is in `ALWAYS_ALLOWED_MODULES`. The `CAPABILITY_MODULES` dict is mutable and shared between `capability_modules` and `ast_scan`. A plugin can mutate the mapping in place to remap arbitrary modules to a default-granted capability.
**Proof of Concept:** Two-plugin chain: plugin A (zero-cap) `import tgirl.plugins.capability_modules as cm; cm.CAPABILITY_MODULES[Capability.CLOCK] = frozenset({"socket", "subprocess", "ctypes"}) | cm.CAPABILITY_MODULES[Capability.CLOCK]`. Plugin B (zero-cap) `import socket; socket.socket()` → succeeds. Both plugins load and execute. Real socket created.
**Client Challenge:** None — accepted at CRITICAL.
**Auditor Defense:** N/A.
**Resolution:** Stands at CRITICAL. The wildcard `tgirl` ALWAYS_ALLOWED list combined with mutable internal state is a single-vector full sandbox escape.
**Remediation:** **Primary fix (structural):** add `tgirl.plugins.{capability_modules,guard,loader,ast_scan,config,errors}` to BANNED_MODULES (or equivalent explicit-reject in Gate 1). Consolidate Gate 1's import classification through a single helper that shares the `root != "tgirl"` exclusion currently only present in `is_allowed_for_grant` (line 220). **Defense-in-depth (partial only):** wrap `CAPABILITY_MODULES` in `types.MappingProxyType` — note this does NOT close the bypass on its own, because `cm.CAPABILITY_MODULES = malicious_dict` (rebinding the module attribute) defeats the proxy. The BANNED additions are the load-bearing fix; the proxy is a hardening layer.
**Effort Estimate:** S.

### Finding 3: `guard_scope` mutates global `sys.modules` (concurrency hazard)
**Initial Severity:** HIGH (CRITICAL once Task 11 wires plugins into uvicorn)
**Final Severity:** MEDIUM (challenged; auditor closed task without defending)
**Category:** Concurrency / race conditions
**Affected Code:** `src/tgirl/plugins/guard.py:54-87` (entry/exit blocks sweep sys.modules).
**Description:** `guard_scope` deletes capability-gated modules from `sys.modules` at scope entry to force finder re-resolution. Other threads observe the wrapper and lose access to the real module's callables. Save/restore between concurrent `guard_scope` blocks can permanently install wrappers (sys.modules pollution).
**Proof of Concept:** Thread A enters `guard_scope`; thread B (unrelated, e.g. uvicorn background task) sees `CapabilityScopedModule` wrapper in `sys.modules['socket']`. Verified.
**Client Challenge:** Severity reflects v1.1 hot-reload concerns and concurrent-load races that v1's server-static plugin model explicitly excludes (PRD §6.5/§7). v1 loads all plugins at startup, single-threaded, then never reloads. The actual current-state bug is an inaccurate docstring claim ("tgirl's own imports pass through" is false during the load window for same-thread imports).
**Auditor Defense:** Auditor closed their task without defending the original HIGH severity. Client challenge stands.
**Resolution:** Downgraded to MEDIUM with a structural-blocker tag for v1.1 hot-reload work. Team lead may choose to defer to the auditor's HIGH if the v1.1 risk model is preferred — both are defensible. The `sys.modules` mutation pattern remains a code-quality concern that should be eliminated.
**Remediation:** Skip `sys.modules` mutation entirely. Have the meta_path finder consult a thread-local registry of currently-active grants. Subsumed by #1's clean-rewrite option.
**Effort Estimate:** S–M.

#### Sub-finding 3.1: Docstring inaccuracy in `guard.py:11-15`
**Severity:** LOW
**Description:** The module docstring claims "tgirl's own imports pass through" — false during the load window for same-thread imports. Misleading for future maintainers reasoning about the contract.
**Remediation:** Rewrite the docstring claim to: *"tgirl's own imports pass through outside `guard_scope`; inside `guard_scope`, fresh imports of capability-gated modules are subject to the plugin's grant — keep all tgirl-internal imports of those modules at module-import time, not lazy/local."*
**Effort Estimate:** XS.

### Finding 4: Sandbox B `__builtins__` never restricted; raw `open()` bypass
**Initial Severity:** CRITICAL
**Final Severity:** CRITICAL
**Category:** Sandbox / privilege escalation
**Affected Code:** `src/tgirl/plugins/loader.py:121-148` (uses `spec.loader.exec_module(module)` with full builtins); `src/tgirl/plugins/ast_scan.py:37-53` (FORBIDDEN_NAMES omits `open`, `breakpoint`, `input`).
**Description:** PRP §Task 4 §Sandbox B safe-builtins contract specifies removal of `open`, `breakpoint`, `input` from plugin module builtins. Implementation never substitutes `module.__builtins__`. AST scan also doesn't include these names in FORBIDDEN_NAMES.
**Proof of Concept:** Zero-capability plugin executes `open('/etc/hosts','r').read()` and `open('/tmp/tgirl_plugin_pwned','w').write(...)` at top level. Loaded successfully; marker file written; verified empirically. Both FILESYSTEM_READ and FILESYSTEM_WRITE capabilities are bypassed — neither was even declared.
**Client Challenge:** None — accepted at CRITICAL.
**Auditor Defense:** N/A.
**Resolution:** Stands at CRITICAL. This single fix collapses Finding #5's blast radius too — computed-string reflection to `__import__` fails when `__import__` is absent from the substituted builtins dict.
**Remediation:** Refactor `_import_plugin_module` so that BOTH file-path and dotted-module branches use the uniform pattern: `find_spec` → `module_from_spec` → `module.__dict__["__builtins__"] = safe_builtins` → `exec_module`. Wire the existing-but-orphaned `capability_open` helper (`capability_modules.py:141-194`, defined with **zero call sites** — strong evidence the implementation was paused mid-Task-6) as the capability-conditional file-open primitive. The uniform pattern eliminates the duplicate code paths AND ensures Sandbox B applies regardless of plugin discovery mode.
**Effort Estimate:** S.

### Finding 5: Full sandbox escape via `getattr` + `__globals__` reflection chain
**Initial Severity:** CRITICAL
**Final Severity:** CRITICAL
**Category:** Sandbox / privilege escalation
**Affected Code:** `src/tgirl/plugins/ast_scan.py:82-99` (`_check_getattr_call` only inspects `Constant` strings); `src/tgirl/plugins/ast_scan.py:37-53` (FORBIDDEN_NAMES omits `getattr`).
**Description:** `_check_getattr_call` only catches `Constant` strings as the second `getattr` argument. Runtime-computed dunder strings (e.g. `chr(95)+chr(95)+"globals"+chr(95)+chr(95)`) plus dict subscription (not attribute access) of `__builtins__["__import__"]` evades all three gates.
**Proof of Concept:** Zero-capability plugin uses computed-string `__globals__` access, walks `_f.__globals__["__builtins__"]["__import__"]("socket")`, creates real socket. Tool call result string `'PWNED-via-getattr-globals-chain'` returned to client. Verified empirically.
**Client Challenge:** None — accepted at CRITICAL.
**Auditor Defense:** N/A.
**Resolution:** Stands at CRITICAL. Three remediations needed because reflection is the canonical pivot point.
**Remediation:** **Primary collapse:** Sandbox B `__builtins__` substitution from #4 — once `__import__` isn't in the substituted builtins dict, the chain raises `KeyError` on `b["__import__"]` and the primary attack path is closed. **Defense-in-depth (after #4):** (a) Add `getattr` to FORBIDDEN_NAMES (revert the narrowing comment in `ast_scan.py:32-36` — its rationale is wrong). (b) Tighten `_check_getattr_call` to reject any non-`Constant` attr-name argument. (c) Add an `ast.Subscript` handler that flags dunder-string subscripts (constant or computed). The `__getattribute__` ban originally considered is unnecessary if #4 lands — withdrawn.
**Effort Estimate:** S.

### Finding 6: Lark grammar terminal injection via unescaped tool name
**Initial Severity:** MEDIUM
**Final Severity:** MEDIUM
**Category:** Input validation
**Affected Code:** `src/tgirl/grammar.py:327, 358, 361`.
**Description:** Tool name is interpolated directly into Lark string-terminal positions without escaping. A name containing quote characters becomes multiple terminals.
**Proof of Concept:** Tool name `evil" "injected` produces grammar terminal `"(" "evil" "injected" SPACE ...` — two terminals instead of one. Could permit grammar drift by malicious or pathological tool names.
**Client Challenge:** Accepted with scope refinement: the impact statement should clarify that valid Python identifiers (per `@tool()` decorator) cannot contain quote characters, but JSON-schema-registered tools via `register_from_schema` accept arbitrary strings.
**Auditor Defense:** Refinement accepted.
**Resolution:** Stands at MEDIUM. The injection vector exists for `register_from_schema` callers.
**Remediation:** Escape `tool.name` for Lark string-terminal context. Validate tool names at registration time with regex `^[a-zA-Z_][a-zA-Z0-9_.-]*$`.
**Effort Estimate:** XS.

### Finding 7: TOML `module` field accepts path traversal
**Initial Severity:** MEDIUM
**Final Severity:** MEDIUM
**Category:** Input validation / path traversal
**Affected Code:** `src/tgirl/plugins/config.py:75-78`; `src/tgirl/plugins/loader.py:92-99`.
**Description:** The `module` field in `tgirl.toml` accepts arbitrary strings. A path-traversal value reaches the loader.
**Proof of Concept:** `module = "../../../tmp/attacker.py"` is accepted by the config parser; loader executes via `spec.loader.exec_module`. Plugin from outside repo can be loaded.
**Client Challenge:** None.
**Auditor Defense:** N/A.
**Resolution:** Stands at MEDIUM.
**Remediation:** Validate `module` is either a dotted Python identifier OR a config-relative path with no `..` escape, no absolute paths.
**Effort Estimate:** XS.

### Finding 8: `breakpoint`/`input` missing from FORBIDDEN_NAMES (PRP §2b spec drift)
**Initial Severity:** MEDIUM
**Final Severity:** MEDIUM
**Category:** Sandbox / spec drift
**Affected Code:** `src/tgirl/plugins/ast_scan.py:37-53`.
**Description:** PRP §Task 4 §2b lists names to reject in plugin function bodies. `breakpoint` and `input` are missing from the implementation. `breakpoint()` invokes the active debugger; `input()` blocks the server thread.
**Proof of Concept:** Static AST scan does not reject these names; if a plugin calls `breakpoint()` from a tool function, server execution halts at PDB.
**Client Challenge:** None.
**Auditor Defense:** N/A.
**Resolution:** Stands at MEDIUM. PRP-specified-but-unimplemented.
**Remediation:** Add `breakpoint`, `input` to FORBIDDEN_NAMES.
**Effort Estimate:** XS.

### Finding 9: `_register_tools` bypasses `ToolRegistry` public API
**Initial Severity:** LOW
**Final Severity:** LOW
**Category:** Code quality
**Affected Code:** `src/tgirl/plugins/loader.py:184-204` (writes through `target._tools`, `target._callables`, `target._sources`).
**Description:** Direct access to `ToolRegistry` private attributes. No current-state bypass of validation, but a refactor of `ToolRegistry` could silently break the loader's invariants.
**Client Challenge:** None.
**Resolution:** Stands at LOW (code smell, no current bypass).
**Remediation:** Add `ToolRegistry.merge_from(other, *, source: str)` public method; reroute `_register_tools` through it.
**Effort Estimate:** S.

### Finding 10: `env_proxy.__contains__` broken through `CapabilityScopedModule` wrapper
**Initial Severity:** LOW
**Final Severity:** LOW
**Category:** Functional bug
**Affected Code:** `src/tgirl/plugins/capabilities/env_proxy.py:21-25`; `src/tgirl/plugins/guard.py:90-173`.
**Description:** Module-level `__contains__` is unreachable when a module is wrapped by `CapabilityScopedModule`. Python's `in` operator uses `type(obj).__contains__`, not attribute lookup. Wrapping a module hides the dunder protocol.
**Proof of Concept:** `"FOO" in env_proxy` (when wrapped) raises `TypeError: argument of type 'CapabilityScopedModule' is not iterable`.
**Client Challenge:** Scope expansion — this is a structural design constraint of the proxy, not just an env_proxy bug. Recommend documenting that proxy modules must use plain method names (no dunder protocols) OR adding `__contains__` etc. to `CapabilityScopedModule`.
**Auditor Defense:** Accepted scope expansion.
**Resolution:** Stands at LOW.
**Remediation:** Document the proxy-design constraint OR add `__contains__` (and other dunder protocols as needed) to `CapabilityScopedModule`. Recommend the documentation route for design clarity.
**Effort Estimate:** XS.

### Finding 11: `fs_write_proxy.rename` per-spec; type annotation drift
**Initial Severity:** INFO
**Final Severity:** INFO
**Category:** Documentation
**Affected Code:** `src/tgirl/plugins/capabilities/fs_write_proxy.py:47-48`.
**Description:** `rename` is exposed per PRP §Y3 specification. Type annotation on the function does not match `os.rename`'s signature.
**Client Challenge:** None — observation only.
**Resolution:** INFO.
**Remediation:** Fix type annotation. Document file-shadowing implications of FILESYSTEM_WRITE capability in plugins.md.
**Effort Estimate:** XS.

### Finding 12: Security test cluster never invokes registered tools
**Initial Severity:** MEDIUM (spawned by client challenge to #1)
**Final Severity:** MEDIUM
**Category:** Test coverage
**Affected Code:** `tests/test_plugins_security.py` — 18 tests; none invoke `reg.get_callable(name)()` after load.
**Description:** Causally responsible for hiding Finding #1 (Phase B). Would have also caught the runtime tail of Finding #5. Every adversarial test asserts a deny path; none assert the corresponding grant-and-success path.
**Client Challenge:** N/A (spawned by client).
**Auditor Defense:** Accepted at MEDIUM.
**Resolution:** Test coverage gap. Pattern issue, not a single-test fix.
**Remediation:** Adopt a project rule: every existing `test_*_denied` requires a sibling `test_*_granted_and_invoked_succeeds` that calls the registered tool. Adds ~6 parametrized invocation tests across NETWORK / SUBPROCESS / FS_READ / FS_WRITE / ENV.
**Effort Estimate:** S.

## What This Audit Did NOT Find

- **No CRITICAL findings outside the sandbox-cluster.** TOML config, CLI flag handling, grammar generation, namespace collision logic — all clean at HIGH+ severity.
- **No HIGH-severity findings survived after challenge.** The two HIGH candidates (#3 sys.modules concurrency, #1 Phase B initial) both downgraded to MEDIUM after evidence-based pushback.
- **No bypass of the three-gate model when correctly implemented.** The Y6 design is sound; findings #2/#4/#5 demonstrate implementation drift, not design failure.
- **No vulnerabilities in the proxy modules' positive surface.** `env_proxy` / `subprocess_proxy` / `fs_read_proxy` / `fs_write_proxy` correctly restrict their named exposures. The exclusions (`symlink_to`, `chmod`, `chown` from fs_write_proxy) hold.
- **No Hy/RestrictedPython sandbox regression.** Sandbox A's existing `_TgirlNodeTransformer` invariants are preserved; only the new `grant` plumbing was added.
- **No type-mro / `__subclasses__` escape under the existing dunder-rejection rules** (verified via auditor adversarial cases for `type.__mro__`, `().__class__.__base__`, `object.__subclasses__`).

## Remediation Priority

**Audit-recommended fix sequencing — bundle steps 1-4 as a single security-fix commit (all S effort):**

1. **#4 — Sandbox B `__builtins__` substitution** (CRITICAL, S). Refactor `_import_plugin_module` so both file-path and dotted-module branches use uniform `find_spec` → `module_from_spec` → builtins injection → `exec_module`. Wire orphaned `capability_open`. **Primary collapse for #4 and #5** — kills the `__import__` terminal sink for the reflection chain.
2. **#2 — Internal-module ban + Gate 1 consolidation** (CRITICAL, S). Add `tgirl.plugins.{capability_modules,guard,loader,ast_scan,config,errors}` to BANNED_MODULES (load-bearing). Consolidate Gate 1 classification through `is_allowed_for_grant`. `MappingProxyType` on `CAPABILITY_MODULES` is defense-in-depth (partial — module-rebinding bypasses it).
3. **#5 — AST tightening** (CRITICAL, S, defense-in-depth after #4 lands). Add `getattr` to FORBIDDEN_NAMES. Tighten `_check_getattr_call` to reject non-`Constant` arg. Add `ast.Subscript` handler for dunder-string subscripts.
4. **#8 — Add `breakpoint`/`input` to FORBIDDEN_NAMES** (MEDIUM, XS). Bundles cleanly with #5's AST work.
5. **#6 — Tool-name validation + Lark escape** (MEDIUM, XS). Bundles cleanly.
6. **#12 — Test-cluster doubling rule** (MEDIUM, S). Every `test_*_denied` requires a sibling `test_*_granted_and_invoked_succeeds`. Adds ~6 parametrized invocation tests. Closes the dev-readiness gap that hid #1.

**Before code review pass (MEDIUM — fix during Tasks 9-14):**

7. **#1 — Phase B wiring** (MEDIUM, S, per PRP §Task 9). Subsumed by clean-rewrite option (no `sys.modules` mutation).
8. **#3 — Eliminate `sys.modules` mutation** (MEDIUM, S, subsumed by #1 option). Fix misleading docstring per sub-finding 3.1.
9. **#7 — TOML `module` field validation** (MEDIUM, XS).

**Before audit closure (LOW/INFO — nice-to-haves):**

10. **#9 — `ToolRegistry.merge_from` public method** (LOW, S).
11. **#10 — `__contains__` on `CapabilityScopedModule` OR proxy-design doc** (LOW, XS). Recommend the doc route.
12. **#11 — `fs_write_proxy.rename` type annotation + plugins.md note** (INFO, XS).

## Cross-Cutting Recommendations

1. **The PRP is sound; the implementation is incomplete.** Findings #4 and #5 are PRP-specified-but-unimplemented. The PRP correctly anticipated Sandbox B's safe-builtins substitution and the `getattr` ban; the implementation drift is what's exploitable. This is an implementation-quality issue, not a design issue. **Do not reopen the PRP.**

2. **Single-source-of-truth pattern recurs.** Three findings (#2, #6, #9) share the same shape: security-relevant validation logic exists in one helper but is reimplemented inline elsewhere with subtle weakening. CLAUDE.md gotcha entry recommended: *"When you write a security-relevant validator, it must be the single source of truth — make it the only callable; do not let any other call site re-implement the check inline."*

3. **Test pattern: dialectic between deny and grant assertions.** Finding #12 surfaces a structural gap. Adopt a project rule: every adversarial test that asserts a deny path also asserts the grant-and-success counterpart. The PRP's adversarial-case battery design is correct; the implementation only ran the deny half.

4. **The two-key model design is fine; only the implementation is broken.** Y6's three-gate separation is structurally sound. Findings #2/#4/#5 don't refute the three-gate design — they refute the current implementation's enforcement of it. After the four CRITICAL fixes (#4, #2, #5, #12), the three-gate model should hold. Re-audit confirms.

## Audit Defensibility

All HIGH+ findings have concrete proof-of-concept demonstrating exploitation, not just describing it. Severity contour reflects realistic v1 threat model. Effort estimates are reasonable (no L/XL inflation). All 12 findings survived the dialectical challenge — none dismissed, none false-positive. One severity downgrade (#3 HIGH → MEDIUM) was challenged and stood without auditor defense; team-lead may choose to defer to the auditor's HIGH if the v1.1 risk model is preferred — both positions are defensible.

**Recommended audit verdict: REJECT.** Re-audit after #4, #2, #5, #12 land. Once those four are fixed, the remaining MEDIUM/LOW/INFO findings are PR-comment-grade and don't block merge.
