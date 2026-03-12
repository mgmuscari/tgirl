# PRP: tgirl.compile — Hy Compilation and Sandboxed Execution

## Source PRD: docs/PRDs/compile.md
## Date: 2026-03-12

## 1. Context Summary

`tgirl.compile` is the execution engine for grammar-constrained Hy s-expressions. It takes model output strings, parses them through Hy, applies two layers of static analysis (Hy AST + Python AST), and runs code in a sandboxed namespace containing only registered tool callables and composition operator implementations.

This is the most security-sensitive module. Three-layer defense-in-depth: grammar prevents invalid expressions at token level, static analysis catches template bugs before execution, sandbox restricts the runtime namespace.

The pipeline: `source string -> hy.read_many() -> Hy AST analysis -> hy.compile() -> Python AST analysis -> compile() -> sandboxed execution`.

## 2. Codebase Analysis

### Relevant existing patterns
- `src/tgirl/types.py:207-216` -- `PipelineError` already defined with `stage`, `tool_name`, `error_type`, `message`, `hy_source`, `position` fields.
- `src/tgirl/registry.py:164-172` -- `get_callable(name)` returns original function for sandbox namespace construction.
- `src/tgirl/registry.py:174-176` -- `names()` returns sorted tool name list for static analysis validation.
- `src/tgirl/registry.py:154-162` -- `get(name)` returns `ToolDefinition` with `timeout` field for per-tool timeout enforcement.
- `src/tgirl/__init__.py` -- Central exports, will need compile additions.
- `pyproject.toml:23` -- `compile = ["hy>=1.0,<2.0", "RestrictedPython>=7.0"]` already declared.

### Conventions to follow
- Frozen Pydantic models for data types
- structlog for logging
- Private helpers prefixed with `_`
- Test classes grouped by feature (`class TestXxx:`)
- `pytest.fixture()` for shared setup
- TDD: RED -> GREEN -> REFACTOR -> COMMIT

### Integration points
- Input: Hy source string + `ToolRegistry` (for name validation and callable lookup)
- Output: execution result or `PipelineError`
- Upstream: `tgirl.grammar` produces the Hy source strings (but compile has no dependency on grammar -- it takes raw strings)
- Downstream: `tgirl.sample` will call compile to run generated expressions

### Hy 1.x API (from research)
- `hy.read_many(source)` -- iterable of Hy model objects
- `hy.compile(tree, module_name)` -- Python `ast.Module`
- Model classes: `hy.models.Expression`, `hy.models.Symbol`, `hy.models.Integer`, `hy.models.Float`, `hy.models.String`, `hy.models.List`, `hy.models.Dict`
- `->`, `let`, `if`, `try`/`except` are native Hy special forms that compile to Python control flow
- `pmap` is NOT a Hy builtin -- must be custom
- Symbol mangling: Hy converts `-` to `_` in symbol names (e.g., `fetch-data` becomes `fetch_data`)

### RestrictedPython role (from research)
- RestrictedPython operates on Python source strings, not pre-made AST
- Cannot directly process Hy-compiled Python AST without ast.unparse() round-trip (wasteful)
- Decision: use as reference pattern, not direct dependency. Build custom AST analyzer based on RestrictedPython's principles. Keep RestrictedPython in deps for future use or as validation cross-check.

## 3. Implementation Plan

**Test Command:** `pytest tests/test_compile.py tests/test_integration_compile.py -v`

### Task 1: Module skeleton and result types

**Files:** `src/tgirl/compile.py`
**Approach:**
- Define `PipelineResult` (frozen Pydantic model): `result: Any`, `hy_source: str`, `execution_time_ms: float`
- Define `CompileConfig` (frozen Pydantic model): `pipeline_timeout: float = 60.0`, `max_depth: int = 50` (max AST nesting depth)
- Import structlog, set up logger
- Stub the main entry point: `run_pipeline(source: str, registry: ToolRegistry, config: CompileConfig | None = None) -> PipelineResult | PipelineError`
- Define the 5 pipeline stages as constants: `STAGE_PARSE`, `STAGE_STATIC_ANALYSIS`, `STAGE_COMPILE`, `STAGE_AST_ANALYSIS`, `STAGE_EXECUTE`

**Tests:** `tests/test_compile.py`
- `TestCompileTypes`: verify models are frozen, fields exist, default config values
- `TestCompileStubs`: verify stub exists and returns PipelineError with NotImplementedError

**Validation:** `pytest tests/test_compile.py -v`

### Task 2: Hy parsing wrapper

**Files:** `src/tgirl/compile.py`
**Approach:**
- Implement `_parse_hy(source: str) -> list | PipelineError`
  - list contains Hy model objects (hy.models.Object subclasses)
- Call `hy.read_many(source)` and collect results into a list
- Catch Hy parse exceptions -> return `PipelineError(stage="parse", ...)`
- Handle empty input -> return `PipelineError`
- Handle `insufficient-resources` expression -- parse succeeds, static analysis should recognize and handle it

**Tests:** `tests/test_compile.py`
- `TestHyParsing`: valid single call parses, valid pipeline parses, invalid syntax returns PipelineError, empty input returns PipelineError, unclosed paren returns PipelineError

**Validation:** `pytest tests/test_compile.py::TestHyParsing -v`

### Task 3: Hy AST static analyzer

**Files:** `src/tgirl/compile.py`
**Approach:**
- Implement `_analyze_hy_ast(trees: list, tool_names: set[str]) -> PipelineError | None`
- Walk the Hy AST recursively, checking:
  - All Expression first elements (call targets) are either:
    - Registered tool names (in `tool_names`, accounting for Hy symbol mangling: `-` to `_`)
    - Composition operator keywords: `->`, `let`, `if`, `try`, `catch`, `pmap`, `insufficient-resources`
  - No `import` or `require` forms
  - No dangerous builtins: `__import__`, `open`, `getattr`, `setattr`, `delattr`
  - No attribute access (`.` as first element of an Expression)
  - No `defn`, `defmacro`, or other definition forms (no recursive definitions)
  - Track let-bound and threading-bound variable names; all Symbol references must resolve to tool names, composition keywords, let-bound names, or literal values
- Return `None` if analysis passes, `PipelineError(stage="static_analysis", ...)` if any check fails

**Tests:** `tests/test_compile.py`
- `TestHyAstAnalysis`:
  - Valid tool call passes analysis
  - Import form rejected
  - Dangerous builtins rejected
  - Attribute access rejected
  - Unregistered function call rejected
  - Let-bound variable references accepted
  - Unresolved variable references rejected
  - Composition operators accepted
  - `insufficient-resources` expression accepted

**Validation:** `pytest tests/test_compile.py::TestHyAstAnalysis -v`

### Task 4: Python AST analyzer (defense-in-depth)

**Files:** `src/tgirl/compile.py`
**Approach:**
- Implement `_analyze_python_ast(tree: ast.Module, tool_names: set[str]) -> PipelineError | None`
- Walk the Python AST using `ast.walk()`:
  - Reject `ast.Import` and `ast.ImportFrom` nodes
  - Reject `ast.Global` and `ast.Nonlocal` nodes
  - Reject `ast.Attribute` where `attr` starts with `__` (dunder access)
  - Reject `ast.Call` where the function is not an `ast.Name` with id in the allowed set (tool names + composition operators + safe builtins like `list`, `dict`)
- Return `None` if analysis passes, `PipelineError(stage="ast_analysis", ...)` if any check fails

**Tests:** `tests/test_compile.py`
- `TestPythonAstAnalysis`:
  - Clean AST from valid tool call passes
  - AST with Import node rejected
  - AST with dunder attribute access rejected
  - AST with Global/Nonlocal rejected
  - AST with unauthorized Call target rejected

**Validation:** `pytest tests/test_compile.py::TestPythonAstAnalysis -v`

### Task 5: Composition operator runtime implementations

**Files:** `src/tgirl/compile.py`
**Approach:**
- Hy's native `->`, `let`, `if`, `try`/`except` compile to Python control flow -- no custom runtime needed for these
- The Hy compiler handles threading macro, local bindings, conditional, error handling natively
- Custom implementation needed only for `pmap`:
  - `_pmap_impl(fns: list, arg: Any) -> list`: takes a list of callables and an argument, applies each fn to arg, returns list of results
  - v1.0: sequential (iterate and call). Thread pool deferred to v1.1.
- Custom implementation for `insufficient-resources`:
  - `_insufficient_resources_impl(reason: str) -> PipelineError`: returns a structured error indicating the model recognized resource exhaustion
- These functions are injected into the sandbox namespace

**Tests:** `tests/test_compile.py`
- `TestCompositionOperators`:
  - Threading `(-> "hello" (greet) (shout))` produces correct result (test via full pipeline since Hy compiles -> natively)
  - Let binding `(let [x (tool1 ...)] (tool2 x))` works correctly
  - Conditional `(if (pred ...) (then ...) (else ...))` works correctly
  - Try/catch `(try (tool ...) (except [e Exception] (fallback ...)))` works correctly
  - Pmap `(pmap [tool1 tool2] arg)` returns list of results
  - `insufficient-resources` returns PipelineError

**Validation:** `pytest tests/test_compile.py::TestCompositionOperators -v`

### Task 6: Sandbox construction

**Files:** `src/tgirl/compile.py`
**Approach:**
- Implement `_build_sandbox(registry: ToolRegistry) -> dict[str, Any]`
  - Start with empty dict
  - Add each registered tool callable: `sandbox[name] = registry.get_callable(name)`
  - Add `pmap` implementation: `sandbox["pmap"] = _pmap_impl`
  - Add `insufficient_resources` handler (with Hy-mangled name)
  - Explicitly set `sandbox["__builtins__"]` to empty dict to prevent builtins access
  - Add safe builtins needed for Hy-generated code if required (e.g., `list`, `dict` constructors -- determined during testing)
- Implement `_run_in_sandbox(code, sandbox: dict) -> Any`
  - Run the compiled code in the sandbox namespace
  - Capture the result of the last expression (implementation detail: may need to modify the AST to assign last expression to a sentinel variable)
  - Handle runtime exceptions -> `PipelineError(stage="execute", ...)`

**Tests:** `tests/test_compile.py`
- `TestSandbox`:
  - Sandbox contains only registered tool names + pmap + insufficient_resources
  - Sandbox has `__builtins__` set to empty dict
  - Attempting to access builtins in sandbox raises error
  - Tool callables in sandbox are the actual registered functions

**Validation:** `pytest tests/test_compile.py::TestSandbox -v`

### Task 7: Timeout enforcement

**Files:** `src/tgirl/compile.py`
**Approach:**
- Implement timeout using `concurrent.futures.ThreadPoolExecutor` (cross-platform, works outside main thread)
- `_run_with_timeout(fn: Callable, timeout: float) -> Any | PipelineError`
  - Submit fn to single-thread executor
  - Call `future.result(timeout=timeout)`
  - Catch `concurrent.futures.TimeoutError` -> `PipelineError(stage="execute", error_type="TimeoutError", ...)`
- Pipeline timeout wraps the entire `run_pipeline()` call
- Per-tool timeouts: wrap individual tool callables with timeout decorators in the sandbox namespace
  - `_wrap_with_timeout(fn: Callable, timeout: float) -> Callable`: returns a wrapper that enforces timeout
  - During sandbox construction, if `registry.get(name).timeout` is set, wrap the callable

**Tests:** `tests/test_compile.py`
- `TestTimeoutEnforcement`:
  - Tool that exceeds per-tool timeout returns PipelineError
  - Pipeline that exceeds overall timeout returns PipelineError
  - Fast execution completes normally
  - Timeout error has correct stage and error_type

**Validation:** `pytest tests/test_compile.py::TestTimeoutEnforcement -v`

### Task 8: Full pipeline assembly

**Files:** `src/tgirl/compile.py`
**Approach:**
- Implement the full `run_pipeline()` function:
  1. Parse: `_parse_hy(source)` -> Hy AST or PipelineError
  2. Static analysis: `_analyze_hy_ast(trees, tool_names)` -> None or PipelineError
  3. Compile: `hy.compile(tree)` -> Python AST, catch errors -> PipelineError
  4. AST analysis: `_analyze_python_ast(ast_tree, tool_names)` -> None or PipelineError
  5. Build sandbox: `_build_sandbox(registry)`
  6. Compile Python AST to bytecode
  7. Run with timeout -> result or PipelineError
  8. Return `PipelineResult(result=..., hy_source=source, execution_time_ms=...)`
- Structured logging at each stage via structlog

**Tests:** `tests/test_compile.py`
- `TestFullPipeline`:
  - Simple tool call: parse -> analyze -> compile -> run -> result
  - Invalid syntax: PipelineError at parse stage
  - Dangerous code: PipelineError at static_analysis stage
  - Tool not found: PipelineError at static_analysis stage
  - Execution error (tool raises): PipelineError at execute stage
  - Each stage failure produces correct `stage` field in PipelineError

**Validation:** `pytest tests/test_compile.py::TestFullPipeline -v`

### Task 9: Integration tests and exports

**Files:** `tests/test_integration_compile.py`, `src/tgirl/__init__.py`
**Approach:**
- Integration tests: create a ToolRegistry with real tools, run various Hy expressions through the full pipeline
- Test the full pipeline with realistic tool registrations:
  - Single tool call with typed arguments
  - Threading pipeline: `(-> arg (tool1) (tool2))`
  - Let bindings: `(let [x (tool1 arg)] (tool2 x))`
  - Conditional: `(if (pred arg) (tool1 arg) (tool2 arg))`
  - Try/catch: `(try (tool ...) (except [e Exception] (fallback ...)))`
  - Pmap: `(pmap [tool1 tool2] arg)`
  - `insufficient-resources` expression
- Test security: attempt sandbox escapes (import, builtins access, attribute chains)
- Update `__init__.py` to export `run_pipeline`, `PipelineResult`, `CompileConfig`

**Tests:** `tests/test_integration_compile.py`
- `TestEndToEndExecution`: realistic tool calls with result verification
- `TestCompositionIntegration`: all composition operators with real tools
- `TestSecurityIntegration`: comprehensive sandbox escape attempts
- `TestErrorStages`: verify each pipeline stage produces correct PipelineError

**Validation:** `pytest tests/test_integration_compile.py -v`

## 4. Validation Gates

```bash
# Syntax/Style
ruff check src/tgirl/compile.py tests/test_compile.py tests/test_integration_compile.py --fix
mypy src/tgirl/compile.py

# Unit Tests
pytest tests/test_compile.py -v

# Integration Tests
pytest tests/test_integration_compile.py -v

# Full suite (ensure no regressions)
pytest tests/ -v --cov=src/tgirl
```

## 5. Rollback Plan

Compile is a new module with no existing consumers. Rollback = delete `src/tgirl/compile.py`, compile tests, and revert `__init__.py` exports. Required for full tier.

## 6. Uncertainty Log

1. **Hy 1.x model class names**: Research suggests `hy.models.Expression`, `hy.models.Symbol`, etc. but exact names may differ. Will be validated during Task 2 when Hy is imported.

2. **Hy composition forms as native special forms**: `->`, `let`, `if`, `try` are native Hy forms compiling to Python control flow. No custom runtime needed. However, the exact Python AST they produce needs validation -- the Python AST analyzer must not reject legitimate Hy-generated patterns. Validated in Task 4.

3. **Result capture from Hy code**: `hy.compile()` produces `ast.Module`. Running a module doesn't automatically return the last expression's value. Need to determine how to capture the result -- likely by modifying the AST to assign the last expression to a sentinel variable, or by using a different compilation mode. Critical implementation detail for Task 8.

4. **RestrictedPython as direct dependency vs. reference**: Currently in deps but not used directly. Custom AST analyzer (Tasks 3-4) replaces RestrictedPython's compile-time restrictions. May remove from deps or keep for future cross-validation.

5. **Symbol mangling**: Hy converts `-` to `_` in symbol names. Tool names registered as Python identifiers (e.g., `fetch_data`) must match what Hy produces after mangling. The static analyzer must account for this mapping.

6. **Pmap sequential vs. parallel**: v1.0 implements pmap as sequential. Thread pool deferred. Functionally a map, but the interface is forward-compatible.

7. **Grammar-compile interface for ModelType**: Grammar produces Hy dict literals for ModelType values. Compile passes raw dicts to tools. If tools expect Pydantic models, the caller must handle reconstruction. Deferred to integration testing.

8. **Hy-generated Python AST patterns**: The Python AST analyzer must account for patterns Hy's compiler generates -- e.g., temporary variables, wrapper functions, or other constructs that look suspicious to a naive AST walker. The allowed-call-target set must include Hy runtime helpers if needed. Discovered empirically during Task 4.
