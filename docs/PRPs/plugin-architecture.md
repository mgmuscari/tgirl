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

### Existing sandbox (`compile.py`)

- `src/tgirl/compile.py:439` — `class _TgirlNodeTransformer(RestrictingNodeTransformer)` with `# type: ignore[misc]` for RestrictedPython missing stubs.
- Rejects: `Import` (line 449), `ImportFrom` (457), `Global` (465), `Nonlocal` (473), dunder attribute access (495).
- Allows: `_hy_*` identifiers (482), non-dunder attribute access without `_getattr_` wrapping (495).
- `src/tgirl/compile.py:800` — the sandboxed bytecode execution site where compiled user code runs inside the restricted environment. Sandbox dict is assembled per-call.

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
- `PluginManifest(name: str, module: str, allow: frozenset[Capability], source_path: Path | None)` — frozen dataclass.
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
- Do NOT load plugins in this task — just capture and propagate intent. Loading is Task 4.
- Wire the parsed list (`list[PluginManifest]`) and the `allow_capabilities: bool` flag into the existing `ctx.registry` setup path, stashing on `ctx.plugin_manifests` and `ctx.allow_capabilities` (or equivalent).

**Tests (RED first):**
- `test_cli_plugin_flag_captured_as_zero_capability_manifest` — `--plugin math` results in a manifest with `allow=frozenset()`.
- `test_cli_plugin_config_path_loads_toml` — `--plugin-config path.toml` loads correctly.
- `test_cli_both_sources_duplicate_name_fails` — duplicate detection across CLI + config.
- `test_cli_allow_capabilities_flag_default_false` — without the flag, `allow_capabilities` is False in ctx.
- `test_cli_allow_capabilities_flag_sets_true` — with the flag, it True.
- `test_cli_auto_discover_tgirl_toml` — if `$CWD/tgirl.toml` exists and no `--plugin-config`, auto-load it.

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
- For v1 (this task), `grant` is `CapabilityGrant.zero()` for every plugin regardless of manifest `allow`. Capability grants are wired in Task 9.
- Module discovery:
  - If `manifest.module` is an importable Python module (e.g. `tgirl.plugins.stdlib.math`), use `importlib.import_module(...)`.
  - If `manifest.module` is a file path (contains `/` or ends in `.py`), use `importlib.util.spec_from_file_location` (mirrors existing `_load_single_module`).
- **Sandboxed import:** wrap the import in a context that restricts to the `CapabilityGrant`. At zero-capability, attempting to `import socket`, `import subprocess`, `import os` (for a function that would enable filesystem access), etc., raises `CapabilityDeniedError` *before* the plugin register(registry) hook runs.
- Sandbox-at-import is the single riskiest part of this task. Options to consider:
  - (a) Use Python import system hooks (`sys.meta_path`) to intercept plugin-module imports and deny forbidden modules. Scoped via `contextvar` so it only applies during the plugin import.
  - (b) Pre-parse the plugin module AST and reject before loading. Reuses `_TgirlNodeTransformer`.
  - Recommend (a) + (b) combined: AST pre-parse rejects obvious violations at parse time; `sys.meta_path` guard catches dynamic imports (`importlib.import_module(...)` inside a function body that runs at import time).
- After import succeeds, call either `module.register(registry)` or merge `module.registry` (same two strategies as `cli.load_tools_from_path`).
- New error type: `CapabilityDeniedError` with structured fields (`capability: Capability`, `caller: str`, `remediation_hint: str`).
- `load_plugin` does NOT execute any tool function — just imports + registers. Call-time capability enforcement is Task 5/6.

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
- Add a `CapabilityGrant` parameter to `_TgirlNodeTransformer.__init__` (currently defaults to zero).
- `visit_Import` and `visit_ImportFrom` become capability-conditional: if the imported module is in the allowed-module-list for *any* granted capability, permit; else reject.
- Allowed-module list is keyed off the capability mapping (defined in Task 6). For Task 5, stub the mapping as an empty dict → no capability unlocks any import → existing behavior preserved at the default.
- The caller of the sandbox (currently `compile_restricted` at line 517) gains an optional `grant: CapabilityGrant | None = None` parameter. `None` → zero grant.
- Wire through `compile.py` public entry points (`compile_hy_source`, etc.) to accept + propagate the grant. Do NOT wire to the server side yet (Task 11).

**Tests (RED first):**
- `test_sandbox_zero_grant_still_rejects_imports` — regression: existing behavior unchanged at default grant.
- `test_sandbox_knob_accepts_optional_grant_arg` — API surface smoke.
- `test_sandbox_grant_cannot_exceed_declared_capabilities` — if I say `CapabilityGrant({NETWORK})`, only network-mapped modules unlock. For Task 5 with the mapping stubbed empty, no modules actually unlock — but the shape is verified.

**Validation:** `pytest tests/test_compile.py -v && mypy src/tgirl/compile.py`

**Commit:** `feat(compile): capability-conditional sandbox relaxation knob`

---

### Task 6: Capability-to-module mapping

**Files:**
- NEW `src/tgirl/plugins/capabilities.py` — the mapping
- `src/tgirl/compile.py` — consume the mapping from Task 5 knob
- NEW `tests/test_plugins_capabilities.py`

**Approach:**
- `CAPABILITY_MODULES: dict[Capability, frozenset[str]]` — maps each capability to the set of stdlib modules (and common third-party imports) it unlocks:
  - `FILESYSTEM_READ` → `{"os.path", "pathlib", "io"}` (reads only; the `open` builtin is handled separately per Uncertainty Log)
  - `FILESYSTEM_WRITE` → `{"os", "shutil", "pathlib", "io"}` (writes) — NOTE: the `open` builtin with `"w"`/`"a"` is a filesystem write; the mapping lists just module-level; call-site enforcement is Task 7 concern where meaningful
  - `NETWORK` → `{"socket", "urllib", "http", "httpx", "requests", "aiohttp"}`
  - `SUBPROCESS` → `{"subprocess", "multiprocessing"}` — shell-execution builtins in the `os` module are handled by keeping `os` restricted unless a capability grants access
  - `ENV` → `{"os"}` scoped to `os.environ`, `os.getenv` at call time — caveat in Task 7
  - `CLOCK` → `{"time", "datetime"}` (granted by default)
  - `RANDOM` → `{"random", "secrets", "uuid"}` (granted by default)
- Each capability is INDEPENDENT — `NETWORK` does not imply `FILESYSTEM_READ`, etc. A plugin needing both must list both.
- Ambiguous builtins like the file-open primitive — the mapping handles module-level imports; call-time enforcement is Task 7 territory where we need AST introspection.
- The mapping is **the single source of truth** for what each capability means.
- Update `compile.py` call to consume the mapping when checking `visit_Import` / `visit_ImportFrom`.
- CLOCK and RANDOM in the default `CapabilityGrant.zero()` set means their modules (`time`, `random`, `uuid`, etc.) always permit — even for stdlib.

**Tests (RED first):**
- For each capability C in the 7: `test_plugin_with_only_{c}_can_import_{c}_module` + `test_plugin_with_only_{c}_CANNOT_import_other_capability_module` (pairwise matrix — use parametrize).
- `test_zero_grant_still_allows_clock_random_modules` — the two defaults.
- `test_capability_mapping_disjoint_modules` — flag any module appearing in two capability sets (sanity; `os` may intentionally appear in multiple — document that).

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
  - Dynamic import via `importlib.import_module("socket")` inside a function body at plugin import time — caught by `sys.meta_path` guard.
  - Dunder escape: `__import__("socket")` — already rejected by existing `check_name` + dunder rule.
  - Conditional import: `try: import socket; except ImportError: pass` — even inside the try block, the AST-level rejection fires.
  - Module alias: `import socket as s` — rejected (alias does not bypass `visit_Import`).
  - From-import: `from socket import socket` — rejected (goes through `visit_ImportFrom`).
  - Globals manipulation: runtime-compiled string containing `import socket` passed to a compile+run primitive — rejected because the compile primitives are not safe builtins in the sandbox.
  - Relative import attempt: `from . import socket` — rejected.
- Document each adversarial case in the test file docstring: what attack, what defense, what test.

**Tests (RED first):**
- Hypothesis: `test_capability_set_closure_property`
- Adversarial cases listed above, each as a named test.

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
- `load_plugin(manifest, registry, *, grant: CapabilityGrant, allow_capabilities: bool) -> None`.
- If `allow_capabilities=False`, `grant` is FORCED to `CapabilityGrant.zero()` regardless of `manifest.allow`.
- If `allow_capabilities=True` AND `manifest.allow` is non-empty, `grant = CapabilityGrant(capabilities=manifest.allow | default_capabilities)`.
- If `allow_capabilities=True` AND `manifest.allow` is empty, `grant = CapabilityGrant.zero()` — no upgrade.
- Log the effective grant at info level (include `plugin_name`, `requested_caps`, `server_flag_allow_capabilities`, `effective_caps`). Auditability.

**Tests (RED first):**
- `test_manifest_allow_network_but_server_flag_absent_loads_zero_cap` — the critical two-key case.
- `test_manifest_allow_network_and_server_flag_present_grants_network` — happy path.
- `test_manifest_empty_allow_ignored_server_flag` — no upgrade.
- `test_grant_upgrade_logged_at_info_level` — audit trail verification.
- `test_function_call_capability_denied_when_grant_insufficient` — runtime enforcement: a `network`-declared plugin loaded under `--no-allow-capabilities`, calling its network function → `CapabilityDeniedError` with plugin + function + capability + remediation.

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
- Grammar impact: dotted names like `math.add` need to tokenize as a single identifier. Verify `llguidance` / `outlines_adapter` handles this (likely yes — many grammars have dotted identifiers). If not, adjust grammar templates.
- System-prompt impact: the `(math.add 1 2)` form is what the model sees. Instructions regen handles this via the registry snapshot.
- Users writing Hy pipelines can then reference `(math.add 1 2)` unambiguously.

**Tests (RED first):**
- `test_stdlib_and_user_plugin_both_named_count_coexist` — stdlib has `count`, user has `count`, registry ends up with `stdlib_namespace.count` and `user_plugin.count` (or however we finalize the scheme).
- `test_duplicate_plugin_name_fails_fast` — same plugin loaded twice.
- `test_grammar_accepts_dotted_tool_names` — grammar parse-check fixture.
- `test_system_prompt_includes_dotted_names_when_namespaced` — instructions output.
- **Decision to finalize in task execution:** Exact naming convention — `<plugin>.<fn>`? `<plugin>/<fn>`? The PRD notes (c) as the recommendation with `<plugin>.<function>`; use dot unless grammar tokenization forbids it.

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
- Add a `source` annotation to each `ToolDefinition` indicating its origin: `"stdlib.math"`, `"user_plugin.foo"`, `"inline"` (for JSON-schema registrations), `"at_tool_kwarg"` (for `--tools <path>`).
- `/telemetry` endpoint exposes `tools` field with `{"name": str, "source": str, "capabilities_granted": list[str]}` per tool. Reviewer at a glance sees what loaded.

**Tests (RED first):**
- `test_serve_starts_with_plugin_config_loads_expected_tools` — config with 2 plugins, server starts, `/v1/models` (or `/telemetry`) reflects both.
- `test_serve_zero_config_still_has_stdlib_math` — default stdlib load.
- `test_serve_stdlib_disable_via_config_removes_it` — `enabled = false` opt-out.
- `test_telemetry_exposes_tool_sources_and_grants` — audit trail visible.
- `test_serve_plugin_with_invalid_config_fails_startup` — fast-fail on bad TOML.

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
- Full test suite: `pytest tests/ -v --tb=short` → MUST pass (baseline after PR #24 was 1123; this PR adds ~30+ new tests, final count should be in the 1160-1180 range).
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

- **Sandbox-at-import implementation choice (Task 4).** Going with `sys.meta_path` guard + AST pre-parse. If `sys.meta_path` proves too leaky (e.g., does not catch every dynamic import path), may need to fall back to full-AST sandboxing for the entire plugin module on import. This could slow down plugin loading significantly. Flagged for security audit.

- **`open` builtin call-site enforcement (Task 6).** The `open` builtin is a safe builtin in the default sandbox but has filesystem implications. A zero-capability plugin calling `open("file.txt", "w")` would write to the filesystem. The capability mapping can reject the `open` builtin for zero-capability plugins, but this removes a commonly-needed primitive for even pure-computation plugins that want to read bundled data files. Decision to make in Task 6: either (a) remove `open` from zero-capability safe builtins, adding it only to `FILESYSTEM_READ`/`FILESYSTEM_WRITE` grants, or (b) keep `open` available but intercept calls at runtime with path+mode checks. Recommend (a) — cleaner, but may require non-trivial refactoring of stdlib plugins.

- **Grammar tokenization of dotted names (Task 10).** Whether `llguidance` handles `math.add` as a single identifier or breaks on the dot. Empirical test required in Task 10. If it breaks, fall back to underscore: `math_add`. The namespace scheme can be adjusted without impacting the trust model.

- **Dynamic import via `importlib.import_module` at plugin call time (post-import).** A plugin loaded with zero capabilities could theoretically call `importlib.import_module("socket")` *from within a function body* at runtime, since the sandbox gates on AST nodes. This attack vector requires the plugin to have access to `importlib`, which is NOT in the safe builtins set. Need to verify during Task 7 adversarial cases and call out explicitly in the security audit.

- **Stdlib scope (Task 8).** Only three placeholder functions in this PRP. The full stdlib is a separate PRD (`stdlib-v1`). This PRP does not adjudicate which functions belong in stdlib — only that the mechanism exists.

- **Telemetry endpoint shape (Task 11).** Exposing `capabilities_granted` in `/telemetry` is useful for audit but exposes which plugins have elevated permissions to any reader of the telemetry endpoint. Consider whether `/telemetry` should be auth-gated — a server with no auth (per Ollama gap analysis) leaks plugin grants to anyone who can hit the endpoint. This is a broader auth question out of scope for this PRP but flagged.

- **Execution under team mode with current hooks.** `block-solo-implementation.sh` will block direct edits on this branch. Execute via `/execute-team`; proposer does the edits, code-reviewer validates each commit.

- **Autodiscovery of stdlib.** Task 8 ships a `math` module. Task 11 auto-loads stdlib. The exact mechanism by which "auto-load stdlib" finds the stdlib packages (iterate `src/tgirl/plugins/stdlib/*.py`? hard-code a list?) is a Task 11 implementation detail; hard-coded list is safer (no filesystem glob at startup, and `enabled = false` in config can selectively disable members).

- **`# type: ignore` budget.** PRP targets zero new ignores. If `RestrictedPython` stubs continue missing things in capability-relaxation code, may need 1–2 additional `[misc]` ignores. Flag in commit messages.
