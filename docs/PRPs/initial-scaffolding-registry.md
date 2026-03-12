# PRP: Initial Project Scaffolding and tgirl.registry

## Source PRD: docs/PRDs/initial-scaffolding-registry.md
## Date: 2026-03-12
## Amended: 2026-03-12 (adopt pydantic, annotated-types, structlog, hypothesis)

## 1. Context Summary

tgirl is a greenfield Python library for grammar-constrained compositional tool calling. This PRP covers two deliverables: (1) project scaffolding — pyproject.toml, package layout, dev tooling — and (2) the complete `tgirl.registry` module, which is the foundation everything else depends on.

The registry uses **pydantic v2** for data modeling, type introspection, and validation, and **annotated-types** for constraint types (`Gt`, `Lt`, etc.). This avoids reimplementing type extraction and schema generation that pydantic already provides.

## 2. Codebase Analysis

### Existing patterns
- No Python code exists yet — `src/` and `tests/` contain only `.gitkeep` files
- `.gitignore` already covers Python artifacts (`__pycache__`, `.mypy_cache`, `.pytest_cache`, etc.)
- AGENTS.md specifies: ruff for lint, mypy for type checking, pytest for tests
- AGENTS.md specifies: tests in `tests/` matching `src/` structure, `test_<condition>_<expected_result>` naming

### Conventions to follow
- 4-space indentation, 88-char max line length
- snake_case functions/variables, PascalCase classes, UPPER_SNAKE_CASE constants
- Google-style docstrings for public APIs
- Frozen Pydantic models (`model_config = ConfigDict(frozen=True)`) for immutable data structures
- `src/` layout (src/tgirl/) — standard Python src layout

### Integration points
- `tgirl.types` — shared Pydantic models imported by registry and all downstream modules
- `tgirl.registry.ToolRegistry.snapshot()` — produces `RegistrySnapshot` consumed by grammar module
- Registry must be importable independently: `from tgirl.registry import ToolRegistry`

## 3. Implementation Plan

**Test Command:** `cd /Users/maddymuscari/ontologi/tgirl && python -m pytest tests/ -v`

### Task 1: Project scaffolding — pyproject.toml and package structure

**Files:**
- Create `tgirl/pyproject.toml`
- Create `tgirl/src/tgirl/__init__.py`
- Create `tgirl/src/tgirl/py.typed` (PEP 561 marker)
- Remove `tgirl/src/.gitkeep`, `tgirl/tests/.gitkeep`
- Create `tgirl/tests/__init__.py`
- Create `tgirl/tests/conftest.py`

**Approach:**
- `pyproject.toml` uses `[build-system]` with setuptools, `[project]` metadata with Python >=3.11
- Core dependencies for registry: `pydantic>=2.0`, `annotated-types>=0.6`, `structlog>=24.0`
- Optional dependency groups for future modules:
  - `grammar = ["jinja2>=3.0"]`
  - `compile = ["hy>=1.0,<2.0", "RestrictedPython>=7.0"]`
  - `transport = ["torch>=2.0", "pot>=0.9"]`
  - `sample = ["outlines>=0.1", "torch>=2.0"]`
  - `serve = ["fastapi>=0.110", "uvicorn>=0.28", "transformers>=4.48"]`
  - `bridge = ["mcp>=1.0"]`
  - `all = [all of the above]`
  - `dev = ["ruff", "mypy", "pytest", "pytest-cov", "hypothesis>=6.0"]`
- `[tool.ruff]` config: line-length 88, target Python 3.11, select standard rules
- `[tool.mypy]` config: strict mode, pydantic plugin enabled
- `[tool.pytest.ini_options]`: testpaths, pythonpath
- `__init__.py` exports `__version__ = "0.1.0"` only

**Tests:**
- `pip install -e ".[dev]"` succeeds
- `python -c "import tgirl; print(tgirl.__version__)"` prints `0.1.0`
- `pytest --collect-only` runs without error (even with no tests yet)
- `ruff check src/ tests/` passes
- `mypy src/` passes

**Validation:**
```bash
cd /Users/maddymuscari/ontologi/tgirl && pip install -e ".[dev]" && python -c "import tgirl; print(tgirl.__version__)" && pytest --collect-only && ruff check src/ tests/ && mypy src/
```

### Task 2: Type representation system (tgirl.types)

**Files:**
- Create `tgirl/src/tgirl/types.py`
- Create `tgirl/tests/test_types.py`

**Approach:**
Use frozen Pydantic models for all data structures. The `TypeRepr` system uses a discriminated union pattern with Pydantic:

```python
from pydantic import BaseModel, ConfigDict
from typing import Literal

class PrimitiveType(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["str", "int", "float", "bool", "none"]
    type_tag: Literal["primitive"] = "primitive"

class ListType(BaseModel):
    model_config = ConfigDict(frozen=True)
    element: "TypeRepr"
    type_tag: Literal["list"] = "list"

class DictType(BaseModel):
    model_config = ConfigDict(frozen=True)
    key: "TypeRepr"
    value: "TypeRepr"
    type_tag: Literal["dict"] = "dict"

class LiteralType(BaseModel):
    model_config = ConfigDict(frozen=True)
    values: tuple[str | int | float | bool, ...]
    type_tag: Literal["literal"] = "literal"

class EnumType(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    values: tuple[str, ...]
    type_tag: Literal["enum"] = "enum"

class OptionalType(BaseModel):
    model_config = ConfigDict(frozen=True)
    inner: "TypeRepr"
    type_tag: Literal["optional"] = "optional"

class UnionType(BaseModel):
    model_config = ConfigDict(frozen=True)
    members: tuple["TypeRepr", ...]
    type_tag: Literal["union"] = "union"

class ModelType(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    fields: tuple["FieldDef", ...]
    type_tag: Literal["model"] = "model"

class AnnotatedType(BaseModel):
    model_config = ConfigDict(frozen=True)
    base: "TypeRepr"
    constraints: tuple["ConstraintRepr", ...]
    type_tag: Literal["annotated"] = "annotated"

class AnyType(BaseModel):
    """Represents typing.Any — unconstrained type. Grammar module decides production semantics."""
    model_config = ConfigDict(frozen=True)
    type_tag: Literal["any"] = "any"

TypeRepr = Annotated[
    PrimitiveType | ListType | DictType | LiteralType | EnumType |
    OptionalType | UnionType | ModelType | AnnotatedType | AnyType,
    Field(discriminator="type_tag"),
]
```

The `type_tag` discriminator enables Pydantic's efficient discriminated union deserialization and gives us JSON Schema generation for free.

For constraints, use `annotated-types` directly:
```python
from annotated_types import Gt, Lt, Ge, Le, MultipleOf

# Stored as a serializable representation in AnnotatedType.constraints
class ConstraintRepr(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["gt", "lt", "ge", "le", "multiple_of"]
    value: int | float
```

Also define:
- `FieldDef(name, type_repr, required, default)` — frozen Pydantic model
- `ParameterDef(name, type_repr, default, has_default)` — frozen Pydantic model
- `ToolDefinition` — frozen Pydantic model per spec §3.3
- `RegistrySnapshot` — frozen Pydantic model per spec §3.3
- `PipelineError` — per spec §5.7 (shared type)
- `TelemetryRecord` — stub definition per spec §8.6

Benefits of Pydantic over custom frozen dataclasses:
- `.model_dump()` / `.model_dump_json()` for serialization (telemetry, logging)
- `.model_json_schema()` for JSON Schema generation (MCP bridge, grammar debugging)
- Discriminated union for `TypeRepr` (efficient, self-documenting)
- Validation on construction (catches bugs early)
- Immutability via `frozen=True` config (same guarantee as frozen dataclass)

**Tests (RED first):**
- `test_primitive_types_are_frozen` — all primitive variants reject mutation
- `test_list_type_nests_correctly` — `ListType(element=PrimitiveType(kind="str"))` constructs
- `test_dict_type_nests_correctly` — DictType with nested types constructs
- `test_union_type_preserves_members` — tuple of members preserved
- `test_optional_type_wraps_inner` — OptionalType constructs
- `test_literal_type_preserves_values` — string and int literals preserved
- `test_model_type_holds_fields` — fields tuple preserved
- `test_annotated_type_holds_constraints` — ConstraintRepr values preserved
- `test_any_type_is_frozen` — AnyType is immutable
- `test_type_repr_discriminated_union` — Pydantic correctly deserializes each variant via type_tag
- `test_type_repr_json_round_trip` — `model_dump_json()` → `model_validate_json()` round-trips for all variants
- `test_type_repr_json_schema` — `.model_json_schema()` produces valid schema for discriminated union
- `test_type_repr_equality` — identical structures compare equal
- `test_type_repr_hashable` — all variants are hashable
- `test_parameter_def_frozen` — ParameterDef is immutable
- `test_tool_definition_frozen` — ToolDefinition is immutable
- `test_tool_definition_json_schema` — ToolDefinition produces valid JSON Schema
- `test_registry_snapshot_frozen` — RegistrySnapshot is immutable
- `test_registry_snapshot_quotas_immutable` — quotas cannot be mutated

**Validation:**
```bash
cd /Users/maddymuscari/ontologi/tgirl && python -m pytest tests/test_types.py -v && ruff check src/tgirl/types.py && mypy src/tgirl/types.py
```

### Task 3: Type extraction engine

**Files:**
- Create `tgirl/src/tgirl/_type_extract.py`
- Create `tgirl/tests/test_type_extract.py`

**Approach:**
A function `extract_type(annotation) -> TypeRepr` that recursively converts Python type annotations to `TypeRepr` Pydantic models. Uses `typing.get_origin`, `typing.get_args`, and `isinstance` checks.

Logic:
- `typing.Any` → `AnyType()`
- `str/int/float/bool` → `PrimitiveType(kind=...)`
- `type(None)` → `PrimitiveType(kind="none")`
- `list[T]` (get_origin is list) → `ListType(element=extract_type(T))`
- `dict[K, V]` (get_origin is dict) → `DictType(key=extract_type(K), value=extract_type(V))`
- `Literal[...]` → `LiteralType(values=args)`
- `Optional[T]` (Union with None) → `OptionalType(inner=extract_type(T))`
- `Union[A, B]` (not Optional) → `UnionType(members=tuple(extract_type(x) for x in args))`
- `Enum` subclass → `EnumType(name=cls.__name__, values=tuple(e.value for e in cls))`
- `Annotated[T, ...]` → scan metadata for `annotated-types` constraint instances (`Gt`, `Lt`, `Ge`, `Le`, `MultipleOf`), convert to `ConstraintRepr`, produce `AnnotatedType`. Non-constraint metadata is ignored.
- Pydantic BaseModel subclass → `ModelType` with fields extracted from `model_fields` (Pydantic v2). Each field's annotation is recursively extracted. Required/optional status from field info.
- Unsupported types → raise `TypeError` with descriptive message

Also: `extract_parameters(func) -> tuple[ParameterDef, ...]` — uses `inspect.signature` and `typing.get_type_hints` to extract all params (skipping `self`/`cls`), validates all have annotations, builds `ParameterDef` for each. Extracts return type separately.

**Tests (RED first):**
- `test_extract_any` — `extract_type(Any)` → `AnyType()`
- `test_extract_dict_str_any` — `extract_type(dict[str, Any])` → `DictType(..., AnyType())`
- `test_extract_str` — `extract_type(str)` → `PrimitiveType(kind="str")`
- `test_extract_int` — `extract_type(int)` → `PrimitiveType(kind="int")`
- `test_extract_float` — `extract_type(float)` → `PrimitiveType(kind="float")`
- `test_extract_bool` — `extract_type(bool)` → `PrimitiveType(kind="bool")`
- `test_extract_none` — `extract_type(type(None))` → `PrimitiveType(kind="none")`
- `test_extract_list_str` — `extract_type(list[str])` → `ListType(element=PrimitiveType(kind="str"))`
- `test_extract_list_nested` — `extract_type(list[list[int]])` → nested ListType
- `test_extract_dict` — `extract_type(dict[str, int])` → DictType
- `test_extract_literal_strings` — `extract_type(Literal["a", "b"])` → LiteralType
- `test_extract_literal_ints` — `extract_type(Literal[1, 2, 3])` → LiteralType
- `test_extract_optional` — `extract_type(Optional[str])` → OptionalType
- `test_extract_union` — `extract_type(Union[str, int])` → UnionType
- `test_extract_enum` — custom Enum class → EnumType with correct values
- `test_extract_annotated_with_gt_lt` — `Annotated[int, Gt(0), Lt(100)]` → AnnotatedType with ConstraintRepr
- `test_extract_annotated_without_constraints` — `Annotated[str, "some metadata"]` → base type only (no AnnotatedType wrapper)
- `test_extract_pydantic_model` — BaseModel subclass → ModelType with correct fields
- `test_extract_pydantic_optional_field` — model with Optional field → correct nested types
- `test_extract_pydantic_nested_model` — model containing another model → recursive ModelType
- `test_extract_unsupported_raises` — `extract_type(bytes)` raises TypeError
- `test_extract_parameters_simple_function` — function with (name: str, count: int) → correct ParameterDef tuple
- `test_extract_parameters_with_defaults` — default values captured
- `test_extract_parameters_missing_annotation_raises` — function with untyped param raises TypeError
- `test_extract_return_type` — return type extracted correctly

**Validation:**
```bash
cd /Users/maddymuscari/ontologi/tgirl && python -m pytest tests/test_type_extract.py -v && ruff check src/tgirl/_type_extract.py && mypy src/tgirl/_type_extract.py
```

### Task 4: ToolRegistry and decorator

**Files:**
- Create `tgirl/src/tgirl/registry.py`
- Create `tgirl/tests/test_registry.py`

**Approach:**
`ToolRegistry` is a mutable class that stores tool definitions and provides snapshot generation. Internally it stores `ToolDefinition` Pydantic models.

```python
class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._callables: dict[str, Callable] = {}  # name → original function

    def tool(
        self,
        *,
        quota: int | None = None,
        cost: float = 0.0,
        cost_budget: float | None = None,
        scope: str | None = None,
        timeout: float | None = None,
        cacheable: bool = False,
        description: str = "",
    ) -> Callable:
        # Returns decorator that:
        # 1. Extracts parameters via extract_parameters(func)
        # 2. Extracts return type via extract_type
        # 3. Validates all params have annotations
        # 4. Creates ToolDefinition (Pydantic model, validated on construction)
        # 5. Stores in self._tools and self._callables
        # 6. Returns func UNMODIFIED (metadata-only decorator)

    def snapshot(
        self,
        *,
        scopes: set[str] | None = None,
        restrict_to: list[str] | None = None,
        cost_budget: float | None = None,
    ) -> RegistrySnapshot:
        # 1. Filter tools by scope (include if tool.scope is None OR tool.scope in scopes)
        # 2. Filter by restrict_to if provided
        # 3. Build quotas dict from tool definitions
        # 4. Return frozen RegistrySnapshot with timestamp

    def get(self, name: str) -> ToolDefinition: ...
    def get_callable(self, name: str) -> Callable: ...
    def names(self) -> list[str]: ...
    def __len__(self) -> int: ...
    def __contains__(self, name: str) -> bool: ...
```

Key invariant: `snapshot()` is deterministic — same registry state + same args → identical snapshot (except timestamp). The snapshot's `tools` tuple is sorted by tool name for determinism.

**Tests (RED first):**
- `test_register_simple_tool` — decorator registers tool, function returned unmodified
- `test_register_tool_extracts_metadata` — quota, cost, scope, etc. stored correctly
- `test_register_tool_extracts_types` — parameter types and return type extracted
- `test_register_duplicate_name_raises` — registering same name twice raises ValueError
- `test_register_missing_annotation_raises` — function with untyped param raises TypeError
- `test_register_missing_return_type_raises` — function with no return type raises TypeError
- `test_decorator_returns_original_function` — decorated function is identical object (not wrapped)
- `test_registered_tool_is_callable` — `registry.get_callable("name")` returns the original function
- `test_registry_len` — `len(registry)` matches registered tool count
- `test_registry_contains` — `"tool_name" in registry` works
- `test_registry_get` — `registry.get("tool_name")` returns ToolDefinition
- `test_registry_get_missing_raises` — KeyError for unknown tool
- `test_registry_names` — returns sorted list of registered tool names
- `test_snapshot_basic` — snapshot contains all tools as frozen RegistrySnapshot
- `test_snapshot_scope_filtering_includes_matching` — tool with matching scope included
- `test_snapshot_scope_filtering_excludes_nonmatching` — tool with wrong scope excluded
- `test_snapshot_scope_filtering_includes_unrestricted` — tool with scope=None always included
- `test_snapshot_scope_none_includes_all` — no scope filter → all tools included
- `test_snapshot_restrict_to` — only named tools appear
- `test_snapshot_restrict_to_unknown_ignored` — unknown names silently ignored
- `test_snapshot_quotas` — quotas dict populated from tool definitions
- `test_snapshot_cost_budget` — cost_remaining set from argument
- `test_snapshot_deterministic` — two snapshots from same state are equal (ignoring timestamp)
- `test_snapshot_tools_sorted_by_name` — tools tuple is alphabetically ordered
- `test_snapshot_is_frozen` — modifying snapshot raises error
- `test_snapshot_serializes_to_json` — `snapshot.model_dump_json()` produces valid JSON
- `test_tool_definition_json_schema_from_snapshot` — tools in snapshot have valid JSON Schema

**Validation:**
```bash
cd /Users/maddymuscari/ontologi/tgirl && python -m pytest tests/test_registry.py -v && ruff check src/tgirl/registry.py && mypy src/tgirl/registry.py
```

### Task 5: Module exports and integration test

**Files:**
- Update `tgirl/src/tgirl/__init__.py` — re-export key public types
- Create `tgirl/tests/test_integration_registry.py`

**Approach:**
- `__init__.py` exports: `ToolRegistry`, `ToolDefinition`, `RegistrySnapshot`, `ParameterDef`, `TypeRepr`, all TypeRepr variants, and `FieldDef`, `ConstraintRepr`
- Re-export `Gt`, `Lt`, `Ge`, `Le`, `MultipleOf` from `annotated-types` for convenience
- Integration test exercises the full workflow: register multiple tools with various type signatures → create snapshot with scope and restriction filtering → verify snapshot contents → test serialization round-trips

**Tests (RED first):**
- `test_full_workflow_register_and_snapshot` — register 3 tools with different scopes, types, quotas. Create snapshot with scope filter. Verify correct tools included, types extracted, quotas set.
- `test_complex_type_signatures` — register tool with `dict[str, list[Optional[int]]]` parameter, verify recursive type extraction
- `test_enum_parameter` — register tool with Enum param, verify EnumType in snapshot
- `test_annotated_constraints` — register tool with `Annotated[int, Gt(0), Lt(100)]` param, verify AnnotatedType with ConstraintRepr
- `test_pydantic_model_parameter` — register tool with BaseModel param, verify ModelType in snapshot
- `test_snapshot_json_round_trip` — `model_dump_json()` → `RegistrySnapshot.model_validate_json()` round-trips correctly
- `test_snapshot_json_schema` — `RegistrySnapshot.model_json_schema()` produces valid schema
- `test_imports_from_package` — `from tgirl import ToolRegistry, ToolDefinition, RegistrySnapshot, Gt, Lt` all work

**Validation:**
```bash
cd /Users/maddymuscari/ontologi/tgirl && python -m pytest tests/ -v && ruff check src/ tests/ && mypy src/
```

## 4. Validation Gates

```bash
# Lint
cd /Users/maddymuscari/ontologi/tgirl && ruff check src/ tests/ --fix

# Type Check
cd /Users/maddymuscari/ontologi/tgirl && mypy src/ --ignore-missing-imports

# Unit Tests
cd /Users/maddymuscari/ontologi/tgirl && python -m pytest tests/ -v --tb=short

# Coverage
cd /Users/maddymuscari/ontologi/tgirl && python -m pytest tests/ -v --cov=src/tgirl --cov-report=term-missing
```

## 5. Rollback Plan

Greenfield project — rollback is deleting the branch. No existing code to break.

## 6. Uncertainty Log

1. **Pydantic model inheritance:** Pydantic v2's discriminated unions require a literal `type_tag` field on each variant. This adds a small per-instance overhead but enables efficient deserialization and automatic JSON Schema generation. Acceptable tradeoff.
2. **`RegistrySnapshot.quotas` type:** Pydantic frozen models prevent field reassignment but `dict` contents are still mutable. Options: (a) use a custom validator that wraps in `MappingProxyType` on construction, (b) accept that Pydantic's `frozen=True` prevents reassignment and document the inner-mutability caveat, (c) use a `tuple[tuple[str, int], ...]` instead of dict. Recommend option (a) for consistency with "safety by construction."
3. **`from __future__ import annotations` interaction:** If user code uses string annotations (PEP 563), `typing.get_type_hints()` resolves them. This should work but may fail if the annotations reference names not in the module's namespace. Tests will cover the happy path; edge cases deferred.
4. **Git monorepo structure:** The repo root is `/Users/maddymuscari/ontologi/`, not `tgirl/`. All file paths in this PRP are relative to the `tgirl/` subdirectory, but git operations must be scoped carefully.
5. **Single-scope per tool:** The spec defines `scope: str | None` (singular) on tool registration. Real authorization systems may require multiple scopes per tool (e.g., a tool needing both `db:write` and `pii:access`). The current design follows the spec literally. If multi-scope is needed, it requires a spec revision to change `scope` to `scopes: frozenset[str] | None` on `ToolDefinition` and adjust filtering to require all tool scopes be present in the authorized set (intersection semantics). Flag for spec review before the grammar module PRP.
6. **Thread safety:** `ToolRegistry` is not thread-safe. Concurrent calls to `tool()` or concurrent `tool()` + `snapshot()` could produce inconsistent state. This is acceptable because tool registration is a startup-time operation — all tools should be registered before the server begins handling requests. The `serve.py` module must enforce this. No locking added.
7. **annotated-types version compatibility:** We use `annotated-types` constraint instances (`Gt`, `Lt`, etc.) directly in the extraction logic. These are simple frozen objects. If their API changes between versions, the extraction code breaks. Pin `>=0.6` (current stable) to mitigate.
