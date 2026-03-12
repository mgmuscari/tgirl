# PRP: tgirl.grammar — Dynamic CFG Generation

## Source PRD: docs/PRDs/grammar.md
## Date: 2026-03-12

## 1. Context Summary

`tgirl.grammar` converts `RegistrySnapshot` objects into context-free grammars that constrain LLM token output to only produce well-formed Hy s-expressions invoking registered tools. This is the core enforcement mechanism — safety by construction at the token level.

The module uses Jinja2 templates to generate grammars in **Lark EBNF format** (the format Outlines consumes via `outlines.generate.cfg()`). Four template files compose into a complete grammar: base structure, per-tool productions, type productions, and composition operators.

Key constraint: grammar output must be **deterministic** — same snapshot (modulo timestamp) produces byte-identical grammar text. This enables grammar diffing for debugging and auditing.

## 2. Codebase Analysis

### Relevant existing patterns
- `src/tgirl/types.py` — All TypeRepr variants (lines 24–140): `PrimitiveType`, `ListType`, `DictType`, `LiteralType`, `EnumType`, `OptionalType`, `UnionType`, `ModelType`, `AnnotatedType`, `AnyType`. Discriminated union via `type_tag` field.
- `src/tgirl/types.py:146-169` — `ParameterDef` and `ToolDefinition` with all metadata fields.
- `src/tgirl/types.py:175-202` — `RegistrySnapshot` with tools, quotas, cost_remaining, scopes.
- `src/tgirl/registry.py:105-152` — `snapshot()` method: sorted tool iteration, scope/restrict_to filtering, quota extraction.
- `src/tgirl/__init__.py` — Central exports, will need grammar additions.
- `pyproject.toml:22` — `grammar = ["jinja2>=3.0"]` already declared.

### Conventions to follow
- Frozen Pydantic models for data types (ConfigDict(frozen=True))
- structlog for logging
- Private helpers prefixed with `_`
- Test classes grouped by feature (`class TestXxx:`)
- `pytest.fixture()` for shared setup

### Integration points
- Input: `RegistrySnapshot` from `tgirl.types`
- Output: Grammar string in Lark EBNF format
- Downstream consumer: `tgirl.sample` (will use grammar to create Outlines CFGGuide)
- Templates loaded via `importlib.resources` from `src/tgirl/templates/`

### Lark EBNF format requirements
- Rules are lowercase: `rule_name: alternative1 | alternative2`
- Terminals are UPPERCASE: `TERMINAL: "literal" | /regex/`
- Start rule: `?start: ...`
- Operators: `?` (optional), `*` (zero+), `+` (one+), `~n` (exactly n), `~n..m` (range)
- String literals: `"quoted"`, regex: `/pattern/`
- Imports: `%import common.NUMBER`, `%ignore WS`
- Must be LALR(1) parseable — no ambiguous lookahead

## 3. Implementation Plan

**Test Command:** `pytest tests/test_grammar.py tests/test_integration_grammar.py -v`

### Task 1: Grammar output types and module skeleton

**Files:** `src/tgirl/grammar.py`
**Approach:**
- Define `GrammarOutput` (frozen Pydantic model): `text: str`, `productions: tuple[Production, ...]`, `snapshot_hash: str`
- Define `Production` (frozen Pydantic model): `name: str`, `rule: str`
- Define `GrammarDiff` (frozen Pydantic model): `added: tuple[Production, ...]`, `removed: tuple[Production, ...]`, `changed: tuple[tuple[Production, Production], ...]`
- Define `GrammarConfig` (frozen Pydantic model): `enumeration_threshold: int = 256` (for AnnotatedType ranges)
- Import structlog, set up logger
- Stub `generate(snapshot, config=None) -> GrammarOutput` and `diff(a, b) -> GrammarDiff`

**Tests:** `tests/test_grammar.py`
- `TestGrammarTypes`: verify models are frozen, fields exist, default config values
- `TestGrammarStubs`: verify stubs exist and raise NotImplementedError or return empty output

**Validation:** `pytest tests/test_grammar.py -v`

### Task 2: Type-to-production converter

**Files:** `src/tgirl/grammar.py`
**Approach:**
- Implement `_type_to_rule(type_repr: TypeRepr, rule_name: str, config: GrammarConfig) -> list[Production]`
- Dispatch on `type_tag` discriminator:
  - `"str"` → `ESCAPED_STRING` terminal (double-quoted with escape handling)
  - `"int"` → `SIGNED_INT` terminal
  - `"float"` → `SIGNED_FLOAT` terminal
  - `"bool"` → `"true" | "false"`
  - `"none"` → `"nil"`
  - `"list"` → `"[" (element (" " element)*)? "]"` with recursive element rule
  - `"dict"` → `"{" (dict_entry (" " dict_entry)*)? "}"` where `dict_entry: ":" key " " value`
  - `"literal"` → enumerated alternatives: `"value1" | "value2" | ...` (strings quoted, numbers/bools as-is)
  - `"enum"` → same as literal over enum values
  - `"optional"` → `inner_rule | "nil"`
  - `"union"` → `member1_rule | member2_rule | ...`
  - `"model"` → `"{" required_fields optional_fields "}"` with Hy dict syntax `":field_name" " " value`
  - `"annotated"` → delegate to base type if no numeric constraints, else enumerate if range ≤ threshold, else use base type production
  - `"any"` → `ESCAPED_STRING | SIGNED_INT | SIGNED_FLOAT | "true" | "false" | "nil"` (permissive primitive union)
- Each recursive type generates additional named productions (e.g., `list_of_int`, `dict_str_float`)
- Rule names are derived deterministically from type structure to ensure deduplication

**Tests:** `tests/test_grammar.py`
- `TestTypeProductions`: one test per TypeRepr variant verifying correct production output
- Test recursive types (list of lists, dict of models)
- Test AnnotatedType enumeration vs. constrained production threshold
- Test deterministic rule naming

**Validation:** `pytest tests/test_grammar.py::TestTypeProductions -v`

### Task 3: Tool-to-production converter

**Files:** `src/tgirl/grammar.py`
**Approach:**
- Implement `_tool_to_rules(tool: ToolDefinition, config: GrammarConfig) -> list[Production]`
- Each tool produces:
  - A `call_{name}` rule: `"(" "{name}" " " args_rule ")"`
  - An `args_{name}` rule encoding parameters in order
  - Required params are mandatory in sequence; optional params use `?` operator
  - Parameter values reference type productions from Task 2
- Handle tools with zero parameters: `"(" "{name}" ")"`
- Handle tools with all-optional parameters

**Tests:** `tests/test_grammar.py`
- `TestToolProductions`: tools with varying parameter counts, required/optional mixes
- Test tool with no parameters
- Test tool with default values (optional params)
- Test tool with complex nested types as parameters

**Validation:** `pytest tests/test_grammar.py::TestToolProductions -v`

### Task 4: Jinja2 template system

**Files:** `src/tgirl/templates/base.cfg.j2`, `src/tgirl/templates/tools.cfg.j2`, `src/tgirl/templates/types.cfg.j2`, `src/tgirl/templates/composition.cfg.j2`, `src/tgirl/grammar.py`
**Approach:**
- Create template directory with `__init__.py` (empty, for `importlib.resources`)
- `base.cfg.j2`: top-level structure
  ```
  ?start: pipeline | single_call | insufficient
  pipeline: "(-> " call (" " call)+ ")"
  single_call: call
  call: {{ tool_alternatives }}
  insufficient: "(insufficient-resources " reason ")"
  reason: "quota-exhausted" | "cost-exceeded" | "scope-denied" | "type-mismatch" | "no-valid-plan"
  {% include "tools.cfg.j2" %}
  {% include "types.cfg.j2" %}
  {% include "composition.cfg.j2" %}
  ```
- `tools.cfg.j2`: iterates over tools, emits per-tool productions
- `types.cfg.j2`: iterates over collected type productions, emits type rules and shared terminals
- `composition.cfg.j2`: composition operator rules (threading, let, if, try/catch, pmap)
- Implement `_load_templates() -> jinja2.Environment` using `importlib.resources`
- Implement `_render_grammar(snapshot, config) -> str` that prepares template context and renders

**Tests:** `tests/test_grammar.py`
- `TestTemplateLoading`: verify templates load without error
- `TestTemplateRendering`: verify rendered output is valid Lark EBNF (parseable structure)
- Test with empty snapshot (no tools) — should produce grammar with only insufficient-resources
- Test with single tool — verify tool appears in call alternatives

**Validation:** `pytest tests/test_grammar.py::TestTemplateLoading tests/test_grammar.py::TestTemplateRendering -v`

### Task 5: Composition operator productions

**Files:** `src/tgirl/templates/composition.cfg.j2`, `src/tgirl/grammar.py`
**Approach:**
- Define grammar rules for each composition operator:
  - Threading: `threading: "(-> " expr (" " threaded_call)+ ")"`
  - Let: `let_expr: "(let [" binding+ "] " body ")"`
  - Conditional: `if_expr: "(if " predicate " " then_branch " " else_branch ")"`
  - Error handling: `try_expr: "(try " body " (catch " var " " fallback "))"`
  - Parallel: `pmap_expr: "(pmap [" call+ "] " arg ")"`
- Composition operators nest: a `call` in a pipeline can be a `let_expr`, an `if_expr` can contain a `pipeline`, etc.
- Update the `call` rule to include composition forms as alternatives

**Tests:** `tests/test_grammar.py`
- `TestCompositionProductions`: verify each operator produces valid grammar rules
- Test nesting: threading containing let, conditional containing pipeline

**Validation:** `pytest tests/test_grammar.py::TestCompositionProductions -v`

### Task 6: Grammar generation and determinism

**Files:** `src/tgirl/grammar.py`
**Approach:**
- Implement the public `generate(snapshot, config=None) -> GrammarOutput`:
  1. Collect all type productions from all tools' parameters and return types
  2. Deduplicate type productions by rule name
  3. Generate tool productions
  4. Render via Jinja2 templates
  5. Compute `snapshot_hash` from snapshot (exclude timestamp for determinism)
  6. Return `GrammarOutput` with text, productions list, and hash
- Ensure determinism: sort all collections, use stable iteration order
- The snapshot already sorts tools by name (registry.snapshot() does `sorted(self._tools)`)

**Tests:** `tests/test_grammar.py`
- `TestGenerate`: full generation from realistic snapshots
- `TestDeterminism`: same snapshot (different timestamps) → identical grammar text
- Test with multiple tools sharing parameter types → deduplication works

**Validation:** `pytest tests/test_grammar.py::TestGenerate tests/test_grammar.py::TestDeterminism -v`

### Task 7: Grammar diffing

**Files:** `src/tgirl/grammar.py`
**Approach:**
- Implement `diff(a: GrammarOutput, b: GrammarOutput) -> GrammarDiff`:
  1. Build dict of `{name: Production}` for each grammar
  2. Added = names in b not in a
  3. Removed = names in a not in b
  4. Changed = names in both where rule text differs (tuple of (old, new))
- Simple set-based comparison on production names + equality check on rule text

**Tests:** `tests/test_grammar.py`
- `TestDiff`: add a tool → diff shows new productions; remove a tool → diff shows removed; change parameter type → diff shows changed
- Test diff of identical grammars → empty diff

**Validation:** `pytest tests/test_grammar.py::TestDiff -v`

### Task 8: Integration tests and exports

**Files:** `tests/test_integration_grammar.py`, `src/tgirl/__init__.py`
**Approach:**
- Integration tests: create a ToolRegistry, register diverse tools (various types, quotas, scopes), take snapshot, generate grammar, verify output structure
- Test the full pipeline: registry → snapshot (with scope filtering, restrict_to) → grammar → diff
- Test edge cases: empty registry, single tool, many tools
- Update `__init__.py` to export `generate`, `diff`, `GrammarOutput`, `GrammarDiff`, `GrammarConfig`

**Tests:** `tests/test_integration_grammar.py`
- `TestRegistryToGrammar`: end-to-end with realistic tool registrations
- `TestScopeFilteringGrammar`: verify scoped snapshots produce different grammars
- `TestGrammarWithQuotas`: verify quota information is accessible (preparation for sampling loop enforcement)

**Validation:** `pytest tests/test_integration_grammar.py -v`

## 4. Validation Gates

```bash
# Syntax/Style
ruff check src/tgirl/grammar.py tests/test_grammar.py tests/test_integration_grammar.py --fix
mypy src/tgirl/grammar.py

# Unit Tests
pytest tests/test_grammar.py -v

# Integration Tests
pytest tests/test_integration_grammar.py -v

# Full suite (ensure no regressions)
pytest tests/ -v --cov=src/tgirl
```

## 5. Rollback Plan

Grammar is a new module with no existing consumers. Rollback = delete `src/tgirl/grammar.py`, `src/tgirl/templates/`, grammar tests, and revert `__init__.py` exports.

## 6. Uncertainty Log

1. **Lark EBNF exact syntax**: Based on Outlines documentation research, but not validated against installed Outlines. Grammar may need syntax adjustments when integrated with `tgirl.sample`. Templates are the right abstraction — they can be updated without changing the generation logic.

2. **Quota enforcement**: This PRP defers stateful quota enforcement to `tgirl.sample`. The grammar includes `insufficient-resources` as an always-available alternative, and `GrammarOutput` carries quota information from the snapshot, but the grammar itself does not expand quota states. This is the spec's "Fallback 2" approach. The grammar handles type safety; the sampler handles quotas. This can be revisited.

3. **Hy s-expression syntax in Lark**: The grammar produces Hy-like syntax as string literals in Lark format. Whether Outlines can efficiently guide generation with these specific grammar rules needs validation in `tgirl.sample` integration. The templates can be adjusted.

4. **AnyType production**: Implemented as permissive primitive union (string | int | float | bool | nil). This may be too broad or too narrow depending on actual usage. The config could be extended to customize this.

5. **Composition operator nesting depth**: Recursive composition rules (pipelines containing let containing if containing pipeline) could produce deep parse trees. Lark handles this, but Outlines' CFGGuide performance with deep recursion is untested.
