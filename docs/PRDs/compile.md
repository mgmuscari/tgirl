# PRD: tgirl.compile — Hy Compilation and Sandboxed Execution

## Status: IMPLEMENTED
## Author: agent (Proposer)
## Date: 2026-03-12
## Branch: feature/compile

## 1. Problem Statement

`tgirl.grammar` produces CFGs that constrain LLM output to valid Hy s-expressions. But those s-expressions are just strings — they need to be parsed, verified safe, compiled, and executed against the registered tools. Without `tgirl.compile`, the grammar produces syntactically valid tool calls that cannot actually be run.

This module is the execution engine and the most security-sensitive component of tgirl. It implements a three-layer defense-in-depth strategy:

1. **Grammar prevention** (upstream) — invalid expressions are inexpressible at the token level
2. **Static analysis** (this module) — catches template bugs or edge cases the grammar missed, at both the Hy AST and Python AST levels
3. **Execution sandbox** (this module) — restricts the runtime namespace so even if analysis misses something, the code can only invoke registered tools

The pipeline: `model output → Hy parse → Hy AST analysis → Python AST compilation → Python AST analysis → sandboxed execution → result`.

## 2. Proposed Solution

Implement `tgirl.compile` with the following components:

1. **Hy parser wrapper** — parse model output strings into Hy AST using `hy.read_many()`
2. **Hy AST static analyzer** — verify all function targets are registered tools or composition operators, no imports, no dangerous builtins, no attribute access, no recursive definitions, all variable references resolve
3. **Python AST compiler** — compile Hy AST to Python AST via `hy.compile()`
4. **Python AST analyzer** — walk AST for defense-in-depth: no Import/ImportFrom, no unauthorized Call targets, no dunder attribute access, no Global/Nonlocal
5. **Execution sandbox** — fresh namespace per execution containing only tool callables and composition operator implementations, no builtins, no modules, no global state
6. **Composition operator runtime** — Python implementations of threading (`->`), let, if, try/catch, pmap that execute within the sandbox
7. **Timeout enforcement** — per-tool and overall pipeline timeouts with structured cancellation
8. **Structured error handling** — errors at every stage produce `PipelineError` with stage, tool name, error type, message, source, and position

## 3. Architecture Impact

### Files created
- `src/tgirl/compile.py` — main module: parse, analyze, compile, sandbox, execute
- `tests/test_compile.py` — unit tests
- `tests/test_integration_compile.py` — integration tests (registry → grammar → compile pipeline)

### Files modified
- `src/tgirl/__init__.py` — export compile module public API
- `pyproject.toml` — may need adjustments if RestrictedPython integration differs from expected

### Dependencies
- `hy>=1.0,<2.0` (already declared in `pyproject.toml` under `[compile]`)
- `RestrictedPython>=7.0` (already declared — may use for sandbox enforcement or as reference, but primary sandbox is namespace restriction)

### Data model
- `PipelineError` already defined in `types.py:207-216` — used for all error returns
- New: `PipelineResult` type for successful execution results (or reuse existing patterns)
- Input: Hy source string + `ToolRegistry` (for callable lookup and name validation)

### Interface with tgirl.registry
- `registry.names()` → list of valid tool names for static analysis
- `registry.get_callable(name)` → tool functions for sandbox namespace
- `registry.get(name)` → tool definitions for timeout lookup

## 4. Acceptance Criteria

1. `tgirl.compile.execute(source, registry)` parses, analyzes, compiles, and executes a Hy s-expression string against registered tools.
2. Single tool calls `(tool_name arg1 arg2)` execute correctly and return the tool's result.
3. All composition operators work: threading `(-> ...)`, let bindings `(let [...] ...)`, conditionals `(if ...)`, error handling `(try ... (catch ...))`, parallel execution `(pmap [...] ...)`.
4. Hy AST static analysis rejects: import forms, dangerous builtins (eval, exec, __import__, compile, open, getattr, setattr, delattr), unregistered function calls, attribute access, unresolved variable references.
5. Python AST analysis rejects: Import/ImportFrom nodes, unauthorized Call targets, dunder attribute access, Global/Nonlocal.
6. Sandbox namespace contains only registered tool callables and composition operator implementations — no builtins, no modules, no global state.
7. Per-tool timeouts and overall pipeline timeout are enforced; timeout produces a structured `PipelineError`.
8. All errors at every stage (parse, static_analysis, compile, ast_analysis, execute) produce a `PipelineError` with the correct `stage` field.
9. The `insufficient-resources` expression is handled gracefully (parsed and recognized, not executed as a tool call).
10. All tests pass under `pytest tests/ -v` with `tgirl[compile]` installed.

## 5. Risk Assessment

- **Hy API stability**: Hy 1.x has stable semantics but internal APIs may vary between minor versions. Need to pin carefully and test `read_many()`, `compile()` behavior.
- **Sandbox escape**: The most critical security risk. If any path allows code execution outside the restricted namespace (e.g., through `__builtins__`, `__class__.__subclasses__()`, or frame inspection), the entire safety model breaks. Defense-in-depth mitigates but doesn't eliminate this risk.
- **RestrictedPython integration**: The spec lists RestrictedPython as a dependency. Need to determine whether it's used as the primary sandbox mechanism (compile-time restriction) or as an additional analysis layer. It may conflict with Hy's compilation path.
- **Composition operator semantics**: Threading, let, if, try, pmap need precise Python implementations that match the Hy s-expression semantics. Getting these wrong creates subtle execution bugs.
- **Timeout enforcement**: Python's `signal.alarm` only works on Unix main thread. Need a cross-platform approach (threading-based timeout or `concurrent.futures`).
- **Hy-to-Python AST mapping**: The compile module must understand how Hy transforms s-expressions into Python AST, since the Python AST analyzer needs to reason about what the original Hy code intended.

## 6. Open Questions

1. Should RestrictedPython be used as the primary sandbox (compile Hy → Python source → RestrictedPython compile → restricted bytecode) or as an additional static analysis layer? The Hy → Python AST path may not be compatible with RestrictedPython's source-level compilation.
2. How does `hy.compile()` handle composition operators like `->`, `let`, `if`? Are these native Hy macros or do we need custom macro definitions?
3. Should `pmap` actually execute in parallel (thread pool) or is sequential execution acceptable for v1.0?
4. What's the interface contract for ModelType values from the grammar? The grammar produces Hy dict literals `{"field_name" value}` — does compile need to reconstruct Pydantic models, or pass raw dicts to tools?

## 7. Out of Scope

- Grammar generation (that's `tgirl.grammar`)
- Logit masking / sampling (that's `tgirl.sample`)
- Quota enforcement at execution time (that's `tgirl.sample`'s responsibility via logit masking)
- Caching of tool results (future feature, not v1.0)
- Remote execution or distributed sandbox
- Support for Hy macros beyond the defined composition operators
