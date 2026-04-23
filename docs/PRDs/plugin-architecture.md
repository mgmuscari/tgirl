# PRD: Plugin Architecture

## Status: DRAFT
## Author: Maddy Muscari (drafted with Claude Opus 4.7)
## Date: 2026-04-23
## Branch: feature/plugin-architecture

## 1. Problem Statement

tgirl's current `@registry.tool()` decorator pattern is effectively internal. A stranger who installs `tgirl` and runs `tgirl serve --model <hf>` gets a capable local server with OpenAI-compat tool calling — but to add their own tools they must clone the repo, write a module, pass `--tools <path>`, and restart the process. No security model for trusted-vs-untrusted plugin code. No concept of a bundled "stdlib" that comes out of the box. No composition story with the in-flight `inference-irq-controller` substrate or the proposed inline-Hy executor.

Two forces make this a v0.2 blocker:

1. **The Ollama gap analysis identifies "inline grammar-switched computation" as tgirl's wedge** (see `docs/analysis/ollama-gap-analysis-2026-04-23.md`). The marquee narrative is: *"a local chatbot that can't miscount R's in strawberry because a verified Hy subroutine is doing the counting."* That narrative needs a disciplined way to describe *what* the model can invoke inline, *what* stdlib is always available, and *what* the user has opted into. The plugin architecture is the answer.

2. **The tool-registry pattern is straining under informal usage.** `serve.py --tools <path>` loads arbitrary Python; `register_from_schema` already exists for JSON-schema registration at request time; the `@tool()` decorator covers startup-time registration; none of these is documented as a public API, versioned, or security-bounded. Each new use case grows the surface in an ad-hoc way.

**Why now:**
- Inline-Hy execution (the launch moat) cannot be specified without first settling what the model is allowed to call.
- The stdlib (math, strings, lists — the demo content) needs a home.
- Security must be *designed into the first version* of the plugin surface. Retrofitting a trust model onto an already-shipped plugin API is orders of magnitude harder than baking it in now.
- v0.2 runway is 2–6 weeks. The plugin architecture is the substrate; it has to ship first.

## 2. Proposed Solution

Formalize tgirl's tool-registration pattern into a **public plugin architecture** with an **explicit capability-based trust model**. The existing `@registry.tool()` decorator continues to work; plugins become the public, configurable, security-bounded wrapper around it.

### Three trust tiers

| Tier | Who provides | Capabilities | Distribution |
|---|---|---|---|
| **stdlib** | bundled with tgirl | zero (pure, I/O-free by construction) | ships with every install |
| **user plugin (default)** | user-specified Python module | zero (sandboxed like stdlib) | `tgirl serve --plugin <name>` |
| **user plugin (with capability grants)** | user-specified module | explicit per-capability grants | `tgirl serve --plugin <name>` + two-key opt-in (see below) |

### Capability model

"I/O" decomposes into named capabilities, each independently grantable. The sandbox in `compile.py` relaxes *only* the capability specifically granted to a plugin — other restrictions remain in force.

| Capability | Default | Risk class | Example use |
|---|---|---|---|
| `filesystem-read` | denied | medium (exfil) | read a local dataset |
| `filesystem-write` | denied | high (tamper) | cache computed results |
| `network` | denied | high (exfil + SSRF) | HTTP API lookup |
| `subprocess` | denied | critical (RCE) | call ffmpeg, imagemagick |
| `env` (read environment variables) | denied | medium (secret leak) | read `API_KEY` for a caller |
| `clock` (real time) | granted by default (debatable) | low | `now()`, `sleep()` |
| `random` (non-deterministic) | granted by default (debatable) | low | `uuid()` |

**Two-key opt-in** for any non-default capability:
- Server-level boot flag: `--allow-capabilities` (or equivalent) — no capability grant takes effect without this
- Per-plugin declaration in config: `allow = ["network", "filesystem-read"]` — the plugin receives only the capabilities explicitly listed

A plugin may be *loaded* without its capabilities being *granted* — e.g. if the user declares a plugin with `allow = ["network"]` but boots `tgirl serve` without `--allow-capabilities`, the plugin loads in zero-capability mode and any function that actually needs the network will fail-loud at call time.

### Stdlib invariant

Every stdlib function must be implementable with zero capabilities. If a proposed stdlib tool requires *any* capability, it is not stdlib — it belongs in a bundled-but-optional plugin pack. This is the rule that keeps stdlib shippable by default with no security surprises.

### Configuration

Plugin config lives in a TOML file (exact location: `tgirl.toml` at repo root, or a `[tool.tgirl.plugins]` section in `pyproject.toml` — see Open Questions). Example:

```toml
[plugins.math]
# stdlib plugin; no config needed — always on

[plugins.weather]
module = "my_weather_plugin"
allow = ["network"]

[plugins.ffmpeg_wrap]
module = "user_ffmpeg"
allow = ["subprocess", "filesystem-read", "filesystem-write"]
```

CLI flags override / supplement config (`--plugin math --plugin my_weather_plugin`) for ad-hoc use.

### Hy function binding

A plugin is a Python module. On load, tgirl:
1. Imports the module inside the sandbox at the plugin's capability tier.
2. Discovers all `@tool()`-decorated callables in the module namespace.
3. Registers each into the session's `ToolRegistry`.
4. Grammar + system prompt regenerate from the updated registry state automatically (no new code path — reuses existing `grammar.py` and `instructions.py`).
5. At call time, the function runs inside the sandbox at its plugin's capability tier.

Plugin functions are constrained to be:
- **Pure by default** — no hidden mutable state. (Decision open — see §6.)
- **Statelessly callable** — arguments in, return value out, no class-instance methods as entry points for v1.
- **Synchronous** — async plugin entry points deferred to v1.1.
- **Type-annotated** — every parameter and return type must be annotated. `@tool()` already enforces this.

### Architectural fit

| Concern | Existing code | Plugin system changes |
|---|---|---|
| Register tool | `registry.py:ToolRegistry` + `@tool()` | Unchanged — plugins call existing API |
| Sandbox execution | `compile.py` RestrictedPython | Adds capability-conditional restriction relaxation |
| Grammar generation | `grammar.py` from registry snapshots | Unchanged — reads registry state |
| System prompt | `instructions.py` from registry | Unchanged — reads registry state |
| Server startup | `serve.py` + `cli.py` | New: read `tgirl.toml`, load configured plugins, enforce capability grants |

### Relationship to the launch roadmap

```
inference-irq-controller   (PRP approved; execute pending)
   ↓
plugin-architecture         ← this PRD
   ↓
stdlib-v1                   (PRD needed — bounded list of pure functions)
   ↓
inline-hy-executor          (PRD needed — grammar pushdown stack on IRQ substrate)
   ↓
apple-silicon-server-mvp    (rework of existing PRD; launch thesis now built on inline-Hy)
```

## 3. Architecture Impact

### New modules / files

- `src/tgirl/plugins/` — new package.
  - `__init__.py` — public `load_plugin(name, module_path, capabilities) -> None` + `PluginManifest` type
  - `loader.py` — imports the plugin module inside the sandbox at the specified capability tier; discovers `@tool()`-decorated callables; registers into `ToolRegistry`
  - `capabilities.py` — the `Capability` enum (`filesystem_read`, `filesystem_write`, `network`, `subprocess`, `env`, `clock`, `random`); capability-to-sandbox-mapping (which imports / builtins each capability unlocks)
  - `config.py` — TOML parser for plugin config, merge logic for CLI overrides
  - `stdlib/` — bundled stdlib plugins. Each is a standalone Python module importable as `tgirl.plugins.stdlib.math`, `tgirl.plugins.stdlib.strings`, etc. Separate PRD for scope (see `stdlib-v1` dependency).

### Modified modules

- `src/tgirl/compile.py` — the sandbox gains a `CapabilityGrant` parameter on its main constructor. Currently the sandbox is binary (restricted vs. full Python); it becomes capability-conditional. The `_TgirlNodeTransformer` gains checks that reject imports of modules outside the granted-capability whitelist. Existing code paths that don't pass a grant continue to get a zero-capability (strictest) sandbox.
- `src/tgirl/serve.py` — `load_inference_context` grows optional `plugin_config_path` parameter. At startup, plugins are loaded into the session's `ToolRegistry` in configured order. The `--allow-capabilities` flag propagates here.
- `src/tgirl/cli.py` — new `--plugin <name>`, `--plugin-config <path>`, `--allow-capabilities` flags.
- `src/tgirl/registry.py` — no structural change. `ToolRegistry` may gain a `source: Literal["inline", "tool-kwarg", "plugin"]` annotation on each `ToolDefinition` for auditability.

### Data model

```python
class Capability(str, Enum):
    FILESYSTEM_READ = "filesystem-read"
    FILESYSTEM_WRITE = "filesystem-write"
    NETWORK = "network"
    SUBPROCESS = "subprocess"
    ENV = "env"
    CLOCK = "clock"
    RANDOM = "random"


@dataclass(frozen=True)
class PluginManifest:
    name: str
    module: str  # Python import path
    allow: frozenset[Capability]  # capabilities requested by the plugin config
    source_path: Path | None  # for file-based plugins; None for installed-package plugins


@dataclass(frozen=True)
class CapabilityGrant:
    """Capabilities the server has actually granted this plugin."""
    capabilities: frozenset[Capability]

    @classmethod
    def zero(cls) -> "CapabilityGrant":
        return cls(frozenset({Capability.CLOCK, Capability.RANDOM}))  # default-granted
```

### Dependencies

No new external dependencies. Uses stdlib `tomllib` (Python 3.11+) for config parsing. Existing `RestrictedPython` infrastructure in `compile.py` is the enforcement backbone.

## 4. Acceptance Criteria

1. **Plugin load from TOML config.** A `tgirl.toml` file declaring `[plugins.math]` causes `math` (the stdlib plugin) to load at `tgirl serve` startup. Its functions appear in the registry, the generated grammar, and the system prompt.
2. **Stdlib plugins always load.** Even without any config file, the stdlib plugin pack is registered automatically. (The user may explicitly disable individual stdlib plugins via `[plugins.<name>] enabled = false`.)
3. **Zero-capability enforcement.** A plugin module that tries to `import socket` (or any other network-capable module) at import time FAILS to load if its `allow` list does not include `network`. Fail-loud, not silent. Same for filesystem, subprocess, env.
4. **Two-key opt-in.** A plugin with `allow = ["network"]` loaded while `tgirl serve` was started *without* `--allow-capabilities` fails to acquire the capability. The plugin loads but any function that attempts a network call raises `CapabilityDeniedError` at call time.
5. **Capability enforcement at call time.** When an I/O-tier plugin function runs, the sandbox relaxes *only* the capabilities in its grant. Attempting a denied capability inside a function (e.g., a `network`-granted plugin tries to write a file) raises at the first denied operation, not silently.
6. **Registry auditability.** `GET /telemetry` (or equivalent) exposes each registered tool's `source` (stdlib / user-plugin-name / inline). A reviewer can see at a glance what's loaded and from where.
7. **No regression on existing `@tool()` workflow.** The pre-existing pattern where a user writes a Python module with `@tool()` decorators and passes `--tools <path>` continues to work unchanged. Those tools load as a zero-capability plugin by default.
8. **Grammar regeneration after plugin load.** A request arriving after `tgirl serve` has loaded plugins uses a grammar that includes the plugin-registered functions. No restart-to-see-tools cycle.
9. **Honest failure modes.**
   - Missing module: `PluginLoadError` with the module path, not a generic `ImportError`.
   - Bad config (invalid TOML, unknown capability name): fails startup fast with a pointed error.
   - Capability requested but not granted: `CapabilityDeniedError` with the function name, the capability, and a remediation hint.
10. **Documentation.** `docs/plugins.md` (or equivalent) documents the trust model, capability list, TOML config format, stdlib pack contents, example user plugin, and the two-key security opt-in with a rationale paragraph.
11. **Hypothesis tests.** The capability-to-sandbox-restriction mapping in `capabilities.py` has Hypothesis-generated tests: for each capability in the enum, a plugin granted only that capability passes its characteristic operation and fails on every other capability's characteristic operation.

## 5. Risk Assessment

- **Sandbox relaxation semantics are load-bearing for security.** Capability grants are not a documentation exercise — they're a real code path in `compile.py`'s restriction machinery. A bug here could let a "network"-granted plugin also read the filesystem. Mitigation: Hypothesis property tests (AC#11) verify each capability grants only its named relaxation. `compile.py` is on CLAUDE.md's strongly-recommended-for-audit list; `/security-audit-team` is expected on this PR.
- **Capability creep.** Once the enum exists, there will be pressure to add new capabilities: `gpu`, `sockets-raw`, `signals`, etc. Mitigation: start tight, explicitly defer proposals that can be composed from existing capabilities. `network` covers HTTP/TCP/UDP for v1; fine-grained splits are v1.1.
- **Plugin authors don't read the security docs.** A naive plugin author may request `subprocess` because "they might need it someday." Mitigation: the `allow` list should feel heavy — the TOML example in docs should show `allow = []` is the default and encourages thinking about what the plugin really needs.
- **Stdlib scope creep.** Every "quality of life" addition to stdlib is an addition that bypasses the plugin config boundary for every user. Mitigation: stdlib scope bounded by separate PRD; changes to stdlib require the same review bar as changes to `compile.py`.
- **Runtime cost of plugin loading.** Each plugin import at startup is a cold Python import. Ten plugins × 500ms each = 5s added startup time. Mitigation: benchmark on real config; lazy-load plugins on first tool-call if startup cost exceeds budget. Not blocking for v1.
- **Interaction with inline-Hy executor.** The inline-Hy PRD (downstream) assumes the grammar at each pushdown-stack level is derived from registered tools. Plugin-registered tools must reach that grammar. Already covered by AC#8 (grammar regen after load) but worth calling out.
- **No per-session plugin override.** v1 is server-static: plugins load at `tgirl serve` startup and stay for the process lifetime. A request can't dynamically enable a plugin. This is a deliberate simplification; revisit only if a concrete use case surfaces.
- **Pydantic / tomllib version interplay.** `tomllib` is 3.11+. If we support 3.11 as a minimum (matching current `pyproject.toml`), no external dep needed. Confirm no accidental 3.10 regression.
- **Private-plugin exfil risk.** A plugin that loads at import time could write state to the network even before any function is called. Mitigation: plugin import happens inside the sandbox at the declared capability tier, so `allow = []` plugins can't do I/O at import either.

## 6. Open Questions

1. **Config file location.** `tgirl.toml` at repo root (dedicated), `[tool.tgirl.plugins]` in `pyproject.toml` (co-located with other Python config), or `~/.config/tgirl/config.toml` (per-user)? Recommend: `tgirl.toml` at repo root as the primary, with CLI flags and env var overrides. `pyproject.toml` section is a nice-to-have for Python-package-distributed plugins.
2. **Naming.** "Plugin" is generic and common; "capability pack" is trendier but worse; "extension" collides with browser extensions mental model; "module" collides with Python's `module`. Recommend: stay with `plugin` — it's what users will search for.
3. **Collision handling.** If stdlib defines `count` and a user plugin defines `count`:
   - (a) Last-loaded wins (surprising)
   - (b) Fail startup (strict, maybe too strict)
   - (c) Namespace by plugin name: `math.count` vs `user.count` (verbose but unambiguous)
   - Recommend: (c) with opt-out. Every tool is referenced by `<plugin>.<function>` in the grammar; bare `<function>` works only when unambiguous, otherwise the grammar requires the qualifier.
4. **Stateful plugins.** v1 excludes class-based / stateful plugins. What's the migration path if a plugin genuinely needs state (e.g., a cache)? Recommend: defer to v1.1 with a `@stateful_plugin` decorator that takes an explicit `state: dict` parameter. Punt for now.
5. **Dynamic plugin reload.** If a user edits a plugin file, must they restart the server? v1: yes. Revisit if feedback pressure surfaces.
6. **`clock` and `random` default grants.** Some pure-computation use cases (deterministic replay for tests) want `random` denied. Making them default-granted in v1 is the pragmatic choice (most plugin authors want these). A `deny = ["random"]` counter-opt mechanism would handle the deterministic-replay case if it becomes real.
7. **Plugin signing / verification.** Not in v1. Users trust the Python modules they point `tgirl serve` at; no cryptographic trust chain. Revisit if a plugin distribution story emerges.
8. **Per-capability rate limits.** `network`-granted plugins could be rate-limited (e.g., max 10 HTTP requests per minute). Not in v1; revisit if abuse surfaces.
9. **Plugin versioning.** v1 loads whatever Python module the user points at. No manifest-declared `tgirl_version = ">=0.2"` compatibility check. Revisit in v1.1 once the plugin API stabilizes.

## 7. Out of Scope

- **Class-based / stateful plugins** — v1.1 if demand surfaces.
- **Async plugin entry points** — v1.1.
- **Dynamic plugin reload without restart** — v1.1.
- **Plugin signing / verification** — post-v0.2.
- **Per-capability rate limits** — post-v0.2 if abuse surfaces.
- **Per-request plugin enable/disable** — the plugin set is server-static for v1.
- **Plugin distribution registry** — tgirl does not build a "plugin marketplace." Users point at Python modules they trust.
- **Cross-process plugin isolation** — plugins run inside the server process. No subprocess isolation for v1; the sandbox is the security boundary.
- **`gpu` / `signals` / fine-grained network sub-capabilities** — defer. `network` covers HTTP/TCP/UDP for v1.
- **Stdlib scope** — the stdlib's member list and semantics are the subject of a **separate PRD** (`stdlib-v1`). This PRD only establishes the *mechanism* by which stdlib plugins load and are shipped. Adding or removing individual stdlib functions is out of scope here.
- **Inline-Hy executor** — a **separate PRD** (`inline-hy-executor`) covers the grammar pushdown stack and mid-generation computation flow. This PRD only guarantees that the plugin system's registry state is the input to that executor's grammar generation.
- **Launch writeup + BFCL evidence gate** — those live in the revised `apple-silicon-server-mvp` PRD once the inline-Hy demo is real.
