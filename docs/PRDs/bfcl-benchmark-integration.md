# PRD: BFCL Benchmark Integration

## Status: DRAFT
## Author: Claude (Proposer) + Maddy Muscari
## Date: 2026-03-13
## Branch: feature/bfcl-benchmark-integration

## 1. Problem Statement

tgirl's grammar-constrained generation achieves 15/15 on our internal showcase, but there's no way to compare against published benchmarks. The Berkeley Function Calling Leaderboard (BFCL v4) is the standard benchmark for tool-calling models — it covers simple, multiple, parallel, and irrelevance categories across 1000+ test cases. Without BFCL scores, tgirl's claims about accuracy and correctness are unverifiable by the community.

Currently, tgirl can only register tools via Python decorators with type annotations. BFCL provides function definitions as JSON schema objects. There's no bridge between these two worlds.

## 2. Proposed Solution

Build a benchmark adapter that:

1. **Dynamically registers BFCL function definitions** into a `ToolRegistry` from JSON schema (the `"function"` field in each test entry). This requires a new `registry.register_from_schema()` method that converts JSON schema type definitions to tgirl's `TypeRepr` system.

2. **Translates tgirl output to BFCL format.** tgirl produces Hy s-expressions like `(calculate_triangle_area 10 5)`. BFCL expects Python-style function calls like `[calculate_triangle_area(base=10, height=5)]`. A format adapter converts between these representations.

3. **Runs the benchmark end-to-end** via a standalone script that reads BFCL test data, runs tgirl inference per entry, writes JSONL result files, and invokes the BFCL AST checker for evaluation. This bypasses BFCL's vLLM/SGLang generation pipeline (which is incompatible with grammar-constrained generation) while using the official evaluation checker.

Start with single-turn categories: `simple_python` (400 entries), `multiple` (200), `parallel` (200), `irrelevance` (200). Java/JS, multi-turn, and agentic categories are out of scope for v1.

## 3. Architecture Impact

### Files/modules affected

- `src/tgirl/registry.py` — New `register_from_schema()` method for dynamic tool registration from JSON schema
- `src/tgirl/bfcl.py` — New module: BFCL adapter (schema conversion, output translation, result file I/O)
- `benchmarks/run_bfcl.py` — Benchmark runner script (reads data, runs inference, writes results, invokes checker)

### Data model changes

- `ToolRegistry` gains `register_from_schema(name, schema, description)` that builds `ToolDefinition` + a stub callable from JSON schema
- No changes to `TypeRepr`, `ToolDefinition`, or `RegistrySnapshot` — the existing type system already supports all JSON schema types BFCL uses (str, int, float, bool, list, dict, optional)

### API changes

- `ToolRegistry.register_from_schema(name: str, parameters: dict, description: str = "") -> None`
- `tgirl.bfcl.convert_sexpr_to_bfcl(hy_source: str, tool_def: ToolDefinition) -> str` — s-expression to Python-style call string
- `tgirl.bfcl.load_test_data(category: str) -> list[dict]` — load BFCL test entries
- `tgirl.bfcl.write_results(results: list[dict], path: Path) -> None` — write JSONL result file

### Dependency additions

- `bfcl-eval` (already installed) — used only for test data loading and evaluation checker
- No new core dependencies

## 4. Acceptance Criteria

1. `registry.register_from_schema()` correctly creates `ToolDefinition` with `ParameterDef` and `TypeRepr` from BFCL JSON schema, including required/optional params and all supported types (string, integer, float, boolean, array, dict).
2. The output translator correctly converts tgirl s-expressions to BFCL Python-style format: `(func arg1 arg2)` → `[func(param1=arg1, param2=arg2)]`, preserving parameter names from the tool definition.
3. For `irrelevance` category, the adapter correctly detects when no tool call was made and outputs an appropriate refusal string.
4. The benchmark runner can process at least the `simple_python` category end-to-end: load data → register tools → run inference → write results → evaluate.
5. Result files match the expected BFCL format: JSONL with `{"id": "...", "result": "..."}` per line.
6. The BFCL AST checker can parse and evaluate tgirl's translated output without errors.
7. Unit tests cover schema-to-TypeRepr conversion for all BFCL parameter types.
8. Unit tests cover s-expression-to-BFCL output translation for single and multiple calls.

## 5. Risk Assessment

- **BFCL function names contain dots** (e.g., `spotify.play`, `triangle_properties.get`). tgirl's grammar and Hy parser may not handle dotted names. Mitigation: sanitize names during registration (replace dots with underscores), map back during output translation.
- **Parallel calls require multiple s-expressions per request.** tgirl currently produces one tool call per constrained generation cycle. Mitigation: for parallel calls, either run multiple cycles or use `max_tool_cycles > 1` in session config. The parallel ground truth is order-independent, so sequential generation is acceptable.
- **BFCL evaluation requires `MODEL_CONFIG_MAPPING` registration.** We can't use `bfcl evaluate` CLI directly. Mitigation: call the AST checker functions directly from Python, bypassing the handler/config system.
- **Some BFCL parameter types may not map cleanly** (e.g., `"type": "dict"` used as Gorilla's convention for objects with properties). Mitigation: map `"dict"` to the appropriate tgirl type based on whether properties are specified.

## 6. Open Questions

1. Should `register_from_schema()` live on `ToolRegistry` or in a separate `tgirl.bfcl` adapter module? (Leaning toward `ToolRegistry` since JSON schema registration is generally useful, not BFCL-specific.)
2. How should we handle BFCL's optional parameters with defaults? tgirl's grammar currently generates all parameters. Should we support optional parameter omission in the grammar, or instruct the model to pass default values?
3. For the `multiple` category where multiple different tools must be called sequentially — should we run separate sessions or handle this within a single multi-cycle session?

## 7. Out of Scope

- Java and JavaScript function call categories (require language-specific parsing)
- Multi-turn categories (require stateful conversation with simulated backends)
- Agentic categories (memory, web search)
- Format sensitivity testing (tgirl uses its own format, not the BFCL format variations)
- Leaderboard submission (we produce scores locally, not for public ranking)
- Performance optimization for benchmark throughput (correctness first)
