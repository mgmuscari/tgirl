# PRP: BFCL Benchmark Integration

## Source PRD: docs/PRDs/bfcl-benchmark-integration.md
## Date: 2026-03-13

## 1. Context Summary

tgirl needs to be benchmarked against the Berkeley Function Calling Leaderboard (BFCL v4) to produce verifiable accuracy scores. BFCL provides function definitions as JSON schema and expects output in Python-style `[func(param=val)]` format. tgirl uses Python decorator-based tool registration and produces Hy s-expressions. We need to bridge these two worlds: dynamically register tools from JSON schema, run grammar-constrained inference, and translate the output to BFCL's expected format. The BFCL evaluation pipeline is decoupled from generation — we produce JSONL result files and call the AST checker directly.

## 2. Codebase Analysis

### Relevant existing patterns

- **Tool registration:** `ToolRegistry.tool()` decorator at `src/tgirl/registry.py:33-104`. Internally calls `_type_extract.extract_parameters()` to convert Python annotations to `TypeRepr`. For schema-based registration, we skip this and build `ParameterDef`/`TypeRepr` directly.

- **Type system:** `src/tgirl/types.py:24-140`. `TypeRepr` is a discriminated union of `PrimitiveType`, `ListType`, `DictType`, `OptionalType`, `EnumType`, etc. Maps directly to JSON schema types:
  - `"string"` → `PrimitiveType(kind="str")`
  - `"integer"` → `PrimitiveType(kind="int")`
  - `"float"` → `PrimitiveType(kind="float")`
  - `"boolean"` → `PrimitiveType(kind="bool")`
  - `"array"` with `"items"` → `ListType(element=...)`
  - `"dict"` with `"properties"` → `ModelType` or `DictType`
  - `"tuple"` → `ListType` (closest match)
  - `"any"` → `AnyType()`
  - `"enum"` (via `"enum"` field) → `EnumType` or `LiteralType`

- **Grammar generation:** `src/tgirl/grammar.py:242-325`. `_tool_to_rules()` emits tool name as a quoted string literal (`"{tool.name}"`). Dotted names (e.g., `spotify.play`) work in grammar literals but NOT in Hy parsing. Must sanitize: `spotify.play` → `spotify_play` for grammar/Hy, map back for BFCL output.

- **Compile/execute:** `src/tgirl/compile.py`. The Hy compiler parses s-expressions and executes via `run_pipeline()`. For BFCL, we don't actually execute the functions (they're stubs) — we need the s-expression text and parameter values to translate to BFCL format.

- **SamplingSession.run_chat():** `src/tgirl/sample.py:458+`. The unified API we'll use for inference. Returns `SamplingResult` with `tool_calls: list[ToolCallRecord]`, where each `ToolCallRecord` has `pipeline: str` (the Hy source) and `result: Any`.

- **BFCL test data:** Installed at `bfcl_eval/data/BFCL_v4_*.json` (JSONL). Each entry has `id`, `question` (nested list of messages), `function` (list of JSON schema definitions). Ground truth at `bfcl_eval/data/possible_answer/`.

- **BFCL evaluation:** `bfcl_eval.eval_checker.ast_eval.ast_checker.ast_checker()` — callable directly without needing `MODEL_CONFIG_MAPPING` registration. `bfcl_eval.model_handler.utils.default_decode_ast_prompting()` parses Python-style function call strings.

### Conventions to follow

- Frozen Pydantic models for data types (per `types.py` pattern)
- `structlog` for logging
- Tests use pytest, TDD mandatory
- Module independence: `tgirl.bfcl` depends on `registry` and `types`, not on `sample` or `grammar`

### Integration points

- `ToolRegistry._tools` dict and `ToolRegistry._callables` dict (for direct registration)
- `ToolDefinition`, `ParameterDef`, `TypeRepr` constructors (for building from schema)
- `SamplingSession.run_chat()` (for inference)
- `bfcl_eval.eval_checker.ast_eval.ast_checker.ast_checker()` (for evaluation)
- `bfcl_eval.model_handler.utils.default_decode_ast_prompting()` (for validating our output format)

## 3. Implementation Plan

**Test Command:** `pytest tests/test_bfcl.py tests/test_registry.py -v`

### Task 1: Add `register_from_schema()` to ToolRegistry

**Files:** `src/tgirl/registry.py`, `tests/test_registry.py`

**Approach:**
Add a method `register_from_schema(name, parameters, description, *, return_type)` that:
1. Converts JSON schema `parameters.properties` to a list of `ParameterDef` using a new `_schema_type_to_repr()` function
2. Creates a `ToolDefinition` with proper `TypeRepr` for each parameter
3. Creates a stub callable that returns `None` (BFCL doesn't execute, just checks AST)
4. Registers both in `self._tools` and `self._callables`

Name sanitization: replace `.` with `_` in tool names (BFCL uses dotted names like `spotify.play`). Store original name in a `_name_map` dict for reverse lookup during output translation.

Schema type mapping:
- `"string"` → `PrimitiveType(kind="str")`
- `"integer"` → `PrimitiveType(kind="int")`
- `"float"` → `PrimitiveType(kind="float")`
- `"boolean"` → `PrimitiveType(kind="bool")`
- `"array"` → `ListType(element=extract_items_type)` or `ListType(element=AnyType())`
- `"dict"` without properties → `DictType(key=PrimitiveType(kind="str"), value=AnyType())`
- `"tuple"` → `ListType(element=AnyType())`
- `"any"` → `AnyType()`
- Property with `"enum"` key → `LiteralType(values=tuple(enum_values))`

Handle `required` vs optional: parameters not in `required` list get `has_default=True`, `default=None`.

**Tests:**
- `test_register_from_schema_simple` — register a function with int and str params, verify ToolDefinition
- `test_register_from_schema_types` — all JSON schema types map to correct TypeRepr
- `test_register_from_schema_optional_params` — required vs optional handling
- `test_register_from_schema_dotted_name` — `spotify.play` → `spotify_play`, original preserved
- `test_register_from_schema_enum` — enum field creates LiteralType
- `test_register_from_schema_array` — array with items type
- `test_register_from_schema_snapshot` — registered tools appear in snapshot

**Validation:** `pytest tests/test_registry.py -v -k schema`

### Task 2: Create `tgirl.bfcl` adapter module — schema loading + output translation

**Files:** `src/tgirl/bfcl.py` (CREATE), `tests/test_bfcl.py` (CREATE)

**Approach:**

Create `src/tgirl/bfcl.py` with:

1. `load_test_data(category: str) -> list[dict]` — loads BFCL JSONL test data for a category
2. `load_ground_truth(category: str) -> list[dict]` — loads ground truth entries
3. `register_bfcl_tools(registry: ToolRegistry, functions: list[dict]) -> dict[str, str]` — registers all functions from a BFCL entry into a registry, returns `{sanitized_name: original_name}` mapping
4. `sexpr_to_bfcl(hy_source: str, registry: ToolRegistry, name_map: dict[str, str]) -> str` — converts tgirl s-expression output to BFCL Python-style format

The output translation (`sexpr_to_bfcl`) is the most critical piece:
- Parse the s-expression to extract function name and positional args
- Look up `ToolDefinition` to get parameter names (positional → named mapping)
- Map sanitized name back to original dotted name
- Format as `[original_name(param1=val1, param2=val2)]`
- Handle string quoting: Hy strings use `"..."`, BFCL expects Python-style `"..."`

For the `irrelevance` category: if `run_chat()` produces no tool calls (freeform only), the result is the raw text output — which will fail BFCL's `decode_ast`, correctly scoring as "no function call" (the desired outcome for irrelevance).

**Tests:**
- `test_load_test_data` — loads simple_python, correct count and fields
- `test_register_bfcl_tools` — registers BFCL function defs, name mapping correct
- `test_sexpr_to_bfcl_simple` — `(calculate_triangle_area 10 5)` → `[calculate_triangle_area(base=10, height=5)]`
- `test_sexpr_to_bfcl_string_args` — `(reverse "hello")` → `[reverse(text="hello")]`
- `test_sexpr_to_bfcl_dotted_name` — `(spotify_play "Taylor Swift" 20)` → `[spotify.play(artist="Taylor Swift", duration=20)]`
- `test_sexpr_to_bfcl_multiple_calls` — multiple s-expressions → `[func1(...), func2(...)]`
- `test_bfcl_output_parseable` — verify translated output parses through `default_decode_ast_prompting`

**Validation:** `pytest tests/test_bfcl.py -v`

### Task 3: Create benchmark runner script

**Files:** `benchmarks/run_bfcl.py` (CREATE)

**Approach:**

Standalone script that:
1. Accepts CLI args: `--model`, `--category` (default: `simple_python`), `--limit` (subset for testing), `--result-dir`
2. Loads the model (MLX) and creates reusable components (embeddings, grammar factory, formatter)
3. For each test entry:
   a. Creates fresh `ToolRegistry` and registers BFCL functions via `register_bfcl_tools()`
   b. Creates fresh `SamplingSession` with the registry
   c. Extracts user question from entry's `question[0]` (first turn)
   d. Calls `session.run_chat(messages)`
   e. Translates output via `sexpr_to_bfcl()` or captures freeform text for irrelevance
   f. Appends `{"id": entry_id, "result": translated_output}` to results
4. Writes results as JSONL to `result/<model_name>/BFCL_v4_<category>_result.json`
5. Runs evaluation via `ast_checker` or `relevance_file_runner` directly

Uses KV-cached forward_fn pattern from showcase example. Includes progress bar via `tqdm`.

**Tests:** Integration test — not unit-testable without a model. Verify script structure and argument parsing.

**Validation:** `python -u benchmarks/run_bfcl.py --model mlx-community/Qwen3.5-0.8B-MLX-4bit --category simple_python --limit 5`

### Task 4: Evaluate and iterate on output format

**Files:** `benchmarks/run_bfcl.py` (MODIFY), `src/tgirl/bfcl.py` (MODIFY if needed)

**Approach:**

Run a small subset of the benchmark and debug:
1. Run 10-20 entries from `simple_python`
2. Check if `default_decode_ast_prompting()` can parse our output
3. Check if `ast_checker()` evaluates correctly against ground truth
4. Fix any format translation issues discovered

This is a debugging/iteration task, not a TDD task. The "test" is running the benchmark itself.

**Validation:** Successful evaluation with non-zero accuracy on `simple_python` subset.

### Task 5: Update exports and wire up

**Files:** `src/tgirl/__init__.py` (MODIFY), `pyproject.toml` (MODIFY if needed)

**Approach:**
- Export key BFCL adapter functions from `tgirl.bfcl`
- Add `bfcl-eval` as optional dependency in pyproject.toml under `[project.optional-dependencies]` benchmark group
- Add `benchmarks/` to project structure

**Tests:** Verify imports work.

**Validation:** `python -c "from tgirl.bfcl import load_test_data, register_bfcl_tools, sexpr_to_bfcl"`

## 4. Validation Gates

```bash
# Unit tests
pytest tests/test_bfcl.py tests/test_registry.py -v

# All tests still pass
pytest tests/ -v

# Type checking
mypy src/tgirl/bfcl.py src/tgirl/registry.py

# Lint
ruff check src/tgirl/bfcl.py src/tgirl/registry.py

# Integration: run benchmark on small subset
PYTHONUNBUFFERED=1 python -u benchmarks/run_bfcl.py --model mlx-community/Qwen3.5-0.8B-MLX-4bit --category simple_python --limit 10
```

## 5. Rollback Plan

All changes are additive:
- `src/tgirl/bfcl.py` is a new module with no dependents
- `register_from_schema()` is a new method on `ToolRegistry`, doesn't affect existing decorator API
- `benchmarks/` is a new directory
- Rollback: delete new files, revert `registry.py` method addition

## 6. Uncertainty Log

1. **BFCL `"dict"` type with properties:** BFCL uses `"type": "dict"` to mean "object with properties" (their own convention). When properties are present, this should map to `ModelType` for accurate grammar generation. When no properties, `DictType` is correct. Need to test both cases.

2. **S-expression parsing for output translation:** We're parsing Hy source strings back to extract function names and args. This is fragile — complex nested expressions or string escaping could break. For BFCL simple/multiple categories (simple flat calls), this should be reliable. May need a proper Hy AST-based approach if issues arise.

3. **Parallel calls:** BFCL parallel category expects multiple function calls per request. tgirl's `max_tool_cycles > 1` should handle this if the model generates multiple delimiter-wrapped s-expressions. But the model may not know to generate multiple calls without specific instruction. May need prompt engineering or system prompt modification.

4. **BFCL evaluation bypass:** We're calling `ast_checker()` directly instead of going through `bfcl evaluate` CLI. This avoids the `MODEL_CONFIG_MAPPING` registration requirement but means we need to replicate some of the runner logic (loading data, matching entries, computing accuracy).

5. **Grammar generation for BFCL schemas with many optional params:** Some BFCL functions have 5-10 optional parameters with defaults. The current grammar's trailing-optional chain may produce very large grammar rules. Performance impact unknown.
