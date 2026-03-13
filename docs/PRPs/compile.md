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

### RestrictedPython role (revised)
- `RestrictingNodeTransformer` is a standard `ast.NodeTransformer` — it operates directly on any `ast.Module`, including one from `hy.compile()`. No `ast.unparse()` round-trip needed.
- Decision: use `RestrictingNodeTransformer` directly as the Python AST security layer in Task 4. May subclass to accommodate Hy-generated patterns and tgirl-specific allowlists.

## 3. Implementation Plan

**Test Command:** `pytest tests/test_compile.py tests/test_integration_compile.py -v`

### Task 1: Module skeleton and result types

**Files:** `src/tgirl/compile.py`
**Approach:**
- Define `PipelineResult` (frozen Pydantic model): `result: Any`, `hy_source: str`, `execution_time_ms: float`
- Define `InsufficientResources` (frozen Pydantic model): `reason: str`, `hy_source: str` — represents the model's intentional signal that it cannot act (per TGIRL.md line 287, `insufficient-resources` is a valid grammar alternative alongside `pipeline` and `single_call`, not an error)
- Define `CompileConfig` (frozen Pydantic model): `pipeline_timeout: float = 60.0`, `max_depth: int = 50` (max AST nesting depth)
- Import structlog, set up logger
- Stub the main entry point: `run_pipeline(source: str, registry: ToolRegistry, config: CompileConfig | None = None) -> PipelineResult | InsufficientResources | PipelineError`
- Define the 5 pipeline stages as constants: `STAGE_PARSE`, `STAGE_STATIC_ANALYSIS`, `STAGE_COMPILE`, `STAGE_AST_ANALYSIS`, `STAGE_EXECUTE`

**Tests:** `tests/test_compile.py`
- `TestCompileTypes`: verify `PipelineResult`, `InsufficientResources`, `CompileConfig` are frozen, fields exist, default config values
- `TestCompileStubs`: verify stub exists and returns PipelineError with NotImplementedError

**Validation:** `pytest tests/test_compile.py -v`

### Task 2: Hy parsing wrapper

**Files:** `src/tgirl/compile.py`
**Approach:**
- Implement `_parse_hy(source: str) -> list | PipelineError`
  - list contains Hy model objects (hy.models.Object subclasses)
- Normalize spec-grammar forms to Hy-native forms before parsing: replace `catch` symbol with `except` (the TGIRL.md spec uses `catch` but Hy 1.x uses `except`). Implemented as `_normalize_hy_source(source: str) -> str`.
- Call `hy.read_many(normalized_source)` and collect results into a list
- Catch Hy parse exceptions -> return `PipelineError(stage="parse", ...)`
- Handle empty input -> return `PipelineError`
- Handle `insufficient-resources` expression -- parse succeeds, static analysis should recognize and handle it

**Tests:** `tests/test_compile.py`
- `TestHyParsing`: valid single call parses, valid pipeline parses, invalid syntax returns PipelineError, empty input returns PipelineError, unclosed paren returns PipelineError, `catch` form normalizes to `except` before parsing

**Validation:** `pytest tests/test_compile.py::TestHyParsing -v`

### Task 3: Hy AST static analyzer

**Files:** `src/tgirl/compile.py`
**Approach:**
- Implement `_analyze_hy_ast(trees: list, tool_names: set[str]) -> PipelineError | None`
- Walk the Hy AST recursively, checking:
  - All Expression first elements (call targets) are either:
    - Registered tool names (in `tool_names`, accounting for Hy symbol mangling: `-` to `_`)
    - Composition operator keywords: `->`, `let`, `if`, `try`, `except`, `catch`, `pmap`, `insufficient-resources`
    - Note: The TGIRL.md spec (line 308) uses `catch` for error handling, but Hy 1.x uses `except`. The grammar emits `catch` per spec. Task 2's parser must normalize `catch` to `except` before Hy parsing (simple string substitution of the symbol). The static analyzer accepts both forms.
  - No `import` or `require` forms
  - No dangerous builtins: `__import__`, `open`, `getattr`, `setattr`, `delattr`
  - Attribute access (`.` syntax) is conditionally allowed per TGIRL.md line 388: reject dunder attribute access (names starting and ending with `__`, e.g., `.__class__`, `.__subclasses__`); allow non-dunder field access (e.g., `.name`, `.result`). **Safety ownership:** the grammar is the primary guard — it constrains attribute names to type-system-derived field names from tool return types (e.g., `ModelType` fields). The static analyzer's dunder check is defense-in-depth against grammar template bugs. Note: some non-dunder attributes are security-sensitive (e.g., `.gi_frame`, `.cr_frame`, `.tb_frame` on generators/coroutines/tracebacks can reach `f_globals`/`f_builtins`). These are unreachable in practice because (a) the grammar only emits field names from registered return types, and (b) tool return types are Pydantic models/primitives, not generators or coroutines. The security audit (recommended for compile.py per CLAUDE.md) should verify this invariant holds. If the grammar's attribute productions are ever loosened beyond type-derived names, this policy must be upgraded from a dunder blocklist to a field-name allowlist.
  - No `defn`, `defmacro`, or other definition forms (no recursive definitions)
  - Track let-bound and threading-bound variable names; all Symbol references must resolve to tool names, composition keywords, let-bound names, or literal values
- Return `None` if analysis passes, `PipelineError(stage="static_analysis", ...)` if any check fails

**Tests:** `tests/test_compile.py`
- `TestHyAstAnalysis`:
  - Valid tool call passes analysis
  - Import form rejected
  - Dangerous builtins rejected
  - Dunder attribute access rejected (e.g., `.__class__`)
  - Non-dunder attribute access accepted (e.g., `.name`)
  - Unregistered function call rejected
  - Let-bound variable references accepted
  - Unresolved variable references rejected
  - Composition operators accepted
  - `insufficient-resources` expression accepted

**Validation:** `pytest tests/test_compile.py::TestHyAstAnalysis -v`

### Task 4: Python AST analyzer (defense-in-depth via RestrictedPython)

**Files:** `src/tgirl/compile.py`
**Approach:**
- Implement `_analyze_python_ast(tree: ast.Module, tool_names: set[str]) -> PipelineError | None`
- Use `RestrictingNodeTransformer` from RestrictedPython directly on the Hy-compiled `ast.Module`:
  ```python
  from RestrictedPython import RestrictingNodeTransformer

  def _analyze_python_ast(tree: ast.Module, tool_names: set[str]) -> PipelineError | None:
      errors: list[str] = []
      transformer = RestrictingNodeTransformer(
          errors=errors, warnings=[], used_names=set()
      )
      transformer.visit(tree)
      if errors:
          return PipelineError(stage="ast_analysis", ...)
      return None
  ```
- `RestrictingNodeTransformer` is a standard `ast.NodeTransformer` — it operates directly on any `ast.Module`, including one from `hy.compile()`. No `ast.unparse()` round-trip needed.
- RestrictedPython's transformer handles: Import/ImportFrom rejection, dunder attribute access prevention, Global/Nonlocal rejection, attribute access guard injection (`_getattr_`, `_getitem_`), print function guarding, iteration guarding
- May need to **subclass `RestrictingNodeTransformer`** to:
  - Allow tool names and composition operators as valid Call targets
  - Relax restrictions that conflict with Hy-generated AST patterns (determined empirically — see Uncertainty Log item 8)
  - Add tgirl-specific allowlists (safe builtins needed by Hy runtime)
  - Handle non-dunder attribute access: RestrictedPython's default transformer injects `_getattr_` guard calls for ALL attribute access. Since the spec (section 5.3) allows non-dunder attribute access and the Hy AST layer already rejects dunders, the subclass should either (a) override `visit_Attribute` to only reject dunders and skip guard injection for non-dunder access, or (b) provide a `_getattr_` implementation in the sandbox that allows non-dunder access. Option (a) is preferred — simpler and avoids sandbox pollution.
- Note: The result capture assignment (`_tgirl_result_ = <expr>`) is injected AFTER this analysis step in the pipeline (Task 8), so it does not need to be allowlisted here

**Tests:** `tests/test_compile.py`
- `TestPythonAstAnalysis`:
  - Clean AST from valid tool call passes (RestrictingNodeTransformer accepts it)
  - AST with Import node rejected
  - AST with dunder attribute access rejected
  - AST with non-dunder attribute access accepted (e.g., `.name` on a result)
  - AST with Global/Nonlocal rejected
  - AST with unauthorized Call target rejected
  - Hy-compiled AST from valid pipeline accepted (test RestrictingNodeTransformer on real Hy output)
  - Hy-compiled AST with injection attempt rejected

**Validation:** `pytest tests/test_compile.py::TestPythonAstAnalysis -v`

### Task 5: Composition operator runtime implementations

**Files:** `src/tgirl/compile.py`
**Approach:**
- Hy's native `->`, `let`, `if`, `try`/`except` compile to Python control flow -- no custom runtime needed for these
- The Hy compiler handles threading macro, local bindings, conditional, error handling natively
- Custom implementation needed only for `pmap`:
  - `_pmap_impl(fns: list, arg: Any) -> list`: takes a list of callables and an argument, applies each fn to arg, returns list of results
  - Note: Hy compiles `(pmap [tool1 tool2] arg)` to `pmap([tool1, tool2], arg)` — symbols resolve to callables in the sandbox namespace, so the signature is correct as-is
  - v1.0: sequential (iterate and call). Thread pool deferred to v1.1.
  - **Error semantics (fail-fast):** If any callable raises, `pmap` stops immediately and re-raises. The caller can wrap pmap in `try`/`catch` for error recovery. This is the simplest correct behavior for v1.0 sequential execution. When v1.1 adds parallel execution, partial-failure semantics (collect results + errors) can be revisited.
  - **Timeout interaction:** Per-tool timeout wrappers (Task 7) are applied during sandbox construction, so each callable in the pmap list is already timeout-wrapped. No special handling needed in `_pmap_impl` itself.
- Custom implementation for `insufficient-resources`:
  - `_insufficient_resources_impl(reason: str) -> InsufficientResources`: returns an `InsufficientResources` instance (not `PipelineError`) — this is the model's intentional signal that it cannot act, not an error condition
- These functions are injected into the sandbox namespace

**Tests:** `tests/test_compile.py`
- `TestCompositionOperators`:
  - Threading `(-> "hello" (greet) (shout))` produces correct result (test via full pipeline since Hy compiles -> natively)
  - Let binding `(let [x (tool1 ...)] (tool2 x))` works correctly
  - Conditional `(if (pred ...) (then ...) (else ...))` works correctly
  - Try/catch `(try (tool ...) (catch e (fallback ...)))` works correctly (parser normalizes `catch` to Hy's `except`)
  - Pmap `(pmap [tool1 tool2] arg)` returns list of results
  - Pmap with one failing tool raises (fail-fast, stops execution)
  - Pmap wrapped in try/catch recovers from tool failure
  - Pmap with timeout-wrapped tools respects per-tool timeouts
  - `insufficient-resources` returns `InsufficientResources` (not `PipelineError`)

**Validation:** `pytest tests/test_compile.py::TestCompositionOperators -v`

### Task 6: Sandbox construction

**Files:** `src/tgirl/compile.py`
**Approach:**
- Implement `_build_sandbox(registry: ToolRegistry) -> dict[str, Any]`
  - Start with empty dict
  - Add each registered tool callable: `sandbox[name] = registry.get_callable(name)`
  - Add `pmap` implementation: `sandbox["pmap"] = _pmap_impl`
  - Add `insufficient_resources` handler (with Hy-mangled name)
  - Add result accumulator: `sandbox["_tgirl_result_"] = None` — sentinel variable for capturing the last expression's value (per TGIRL.md spec section 5.5)
  - Explicitly set `sandbox["__builtins__"]` to empty dict to prevent builtins access
  - Add safe builtins needed for Hy-generated code if required (e.g., `list`, `dict` constructors -- determined during testing)
- Implement `_run_in_sandbox(code, sandbox: dict) -> Any`
  - Run the compiled bytecode in the sandbox namespace
  - Retrieve the result from the sentinel: `sandbox["_tgirl_result_"]` (the AST is rewritten in Task 8 to assign the last expression to this variable)
  - Handle runtime exceptions -> `PipelineError(stage="execute", ...)`

**Tests:** `tests/test_compile.py`
- `TestSandbox`:
  - Sandbox contains only registered tool names + pmap + insufficient_resources + `_tgirl_result_` sentinel
  - Sandbox `_tgirl_result_` is initialized to None
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
- Implement `_inject_result_capture(tree: ast.Module) -> ast.Module`:
  - Rewrite the last expression statement in the module body to `_tgirl_result_ = <expr>`
  - Specifically: if the last statement is an `ast.Expr`, replace it with `ast.Assign(targets=[ast.Name(id='_tgirl_result_')], value=expr_node)`
  - Call `ast.fix_missing_locations(tree)` after rewriting
  - This runs AFTER `_analyze_python_ast` so the injected assignment is not subject to security analysis (it is trusted code injected by the engine, not by the model)
- Implement the full `run_pipeline()` function:
  1. Parse: `_parse_hy(source)` -> Hy AST or PipelineError
  2. Static analysis: `_analyze_hy_ast(trees, tool_names)` -> None or PipelineError
  3. Compile: `hy.compile(tree)` -> Python AST, catch errors -> PipelineError
  4. AST analysis: `_analyze_python_ast(ast_tree, tool_names)` -> None or PipelineError
  5. Inject result capture: `_inject_result_capture(ast_tree)` -> rewritten AST
  6. Build sandbox: `_build_sandbox(registry)`
  7. Compile Python AST to bytecode
  8. Run with timeout -> sandbox["_tgirl_result_"] or PipelineError
  9. Check result type: if `isinstance(result, InsufficientResources)`, return it directly
  10. Return `PipelineResult(result=sandbox["_tgirl_result_"], hy_source=source, execution_time_ms=...)`
- Structured logging at each stage via structlog

**Tests:** `tests/test_compile.py`
- `TestResultCapture`:
  - `_inject_result_capture` rewrites last Expr to assignment
  - Module with single expression captures its value
  - Module with multiple statements captures only the last expression
  - Empty module body is handled gracefully
- `TestFullPipeline`:
  - Simple tool call: parse -> analyze -> compile -> run -> result (result correctly captured via sentinel)
  - Invalid syntax: PipelineError at parse stage
  - Dangerous code: PipelineError at static_analysis stage
  - Tool not found: PipelineError at static_analysis stage
  - Execution error (tool raises): PipelineError at execute stage
  - Each stage failure produces correct `stage` field in PipelineError
  - `insufficient-resources` expression returns `InsufficientResources` (not `PipelineResult` or `PipelineError`)

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
- Update `__init__.py` to export `run_pipeline`, `PipelineResult`, `InsufficientResources`, `CompileConfig`

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

3. **Result capture from Hy code** (RESOLVED): `hy.compile()` produces `ast.Module`. Running a module doesn't return the last expression's value. Resolution: `_inject_result_capture()` in Task 8 rewrites the last `ast.Expr` statement to `_tgirl_result_ = <expr>`. The sandbox (Task 6) initializes `_tgirl_result_` to `None`. After execution, the result is read from `sandbox["_tgirl_result_"]`. The injection runs after AST security analysis so it is not subject to RestrictedPython checks.

4. **RestrictedPython integration via RestrictingNodeTransformer**: `RestrictingNodeTransformer` is used directly on the Hy-compiled `ast.Module` — it is a standard `ast.NodeTransformer`, not limited to source strings. May need a custom subclass to accommodate Hy-generated patterns and tgirl tool allowlists. The subclass boundary will be determined empirically during Task 4 implementation.

5. **Symbol mangling**: Hy converts `-` to `_` in symbol names. Tool names registered as Python identifiers (e.g., `fetch_data`) must match what Hy produces after mangling. The static analyzer must account for this mapping.

6. **Pmap sequential vs. parallel**: v1.0 implements pmap as sequential. Thread pool deferred. Functionally a map, but the interface is forward-compatible.

7. **Grammar-compile interface for ModelType**: Grammar produces Hy dict literals for ModelType values. Compile passes raw dicts to tools. If tools expect Pydantic models, the caller must handle reconstruction. Deferred to integration testing.

8. **Hy-generated Python AST patterns vs. RestrictedPython defaults**: Now elevated in importance. RestrictedPython's default `RestrictingNodeTransformer` policy may reject legitimate Hy-generated patterns (e.g., temporary variables, wrapper functions, attribute access patterns) that a custom analyzer would have allowed. Task 4 testing must specifically cover real Hy-compiled AST to discover which restrictions need relaxing via subclass overrides. This is the primary integration risk for the RestrictedPython approach.
