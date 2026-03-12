# PRD: Initial Project Scaffolding and tgirl.registry

## Status: IMPLEMENTED
## Author: proposer (agent)
## Date: 2026-03-12
## Branch: feature/initial-scaffolding-registry

## 1. Problem Statement

tgirl exists as a comprehensive technical specification (TGIRL.md) with full methodology infrastructure (Push Hands) but zero implementation code. Before any module can be built, the project needs a proper Python package structure, dependency management, dev tooling configuration, and a test framework.

The registry module (`tgirl.registry`) is the foundation of the entire dependency graph — grammar, compile, sample, and bridge all depend on it. Nothing else can proceed until tools can be registered, typed, snapshotted, and filtered.

## 2. Proposed Solution

Two deliverables in one feature:

**Scaffolding:** A complete Python project skeleton — `pyproject.toml` with dependency groups, package structure under `src/tgirl/`, dev tool configuration (ruff, mypy, pytest), and CI-ready test infrastructure.

**Registry module:** The full `tgirl.registry` module implementing:
- `@registry.tool()` decorator with metadata extraction (quota, cost, scope, timeout, cacheable, description)
- Type extraction from Python type hints via `inspect` and `typing.get_type_hints`
- Type representation system covering all spec types (str, int, float, bool, None, list[T], dict[K,V], Literal, Enum, Optional, Union, Pydantic BaseModel, Annotated with range constraints)
- Immutable `RegistrySnapshot` generation with quota state, cost budget, and scope/restriction filtering
- `ToolDefinition` and `ParameterDef` frozen dataclasses
- Shared type definitions in `tgirl.types`

## 3. Architecture Impact

### Files created
- `pyproject.toml` — project metadata, dependencies, extras, tool config
- `src/tgirl/__init__.py` — package init with version
- `src/tgirl/types.py` — shared type definitions (TypeRepresentation, ParameterDef, ToolDefinition, RegistrySnapshot, etc.)
- `src/tgirl/registry.py` — ToolRegistry class, tool decorator, snapshot generation
- `tests/conftest.py` — shared fixtures
- `tests/test_registry.py` — comprehensive registry tests
- `tests/test_types.py` — type representation tests

### Files modified
- None (greenfield)

### Dependencies added (core)
- Python >=3.11
- No third-party deps for registry (stdlib only per dependency graph)

### Dependencies added (dev)
- ruff
- mypy
- pytest, pytest-cov

### Dependency groups defined (not installed yet)
- core: jinja2, hy, outlines, torch, scipy, transformers
- serve: fastapi, uvicorn
- bridge: mcp-sdk
- dev: ruff, mypy, pytest, pytest-cov

## 4. Acceptance Criteria

1. `pip install -e ".[dev]"` succeeds and installs the package in editable mode
2. `from tgirl.registry import ToolRegistry, tool` imports without error
3. `from tgirl.types import ToolDefinition, RegistrySnapshot, ParameterDef, TypeRepresentation` imports without error
4. Decorating a function with `@registry.tool()` registers it with all metadata extracted
5. Functions without complete type annotations raise `TypeError` at registration time
6. All Python types in TGIRL.md §3.2 are correctly extracted and represented (str, int, float, bool, None, list[T], dict[K,V], Literal, Enum, Optional, Union, Pydantic BaseModel, Annotated with range constraints)
7. `registry.snapshot()` returns a frozen `RegistrySnapshot` with correct tool definitions, quota state, and cost budget
8. Scope filtering: `snapshot(scopes={"db:write"})` excludes tools requiring scopes not in the provided set; tools with `scope=None` are always included
9. Tool restriction: `snapshot(restrict_to=["tool_a", "tool_b"])` includes only named tools
10. Same registry state produces identical snapshots (deterministic)
11. `ruff check src/ tests/` passes with zero violations
12. `mypy src/` passes with zero errors
13. `pytest tests/ -v` passes with all tests green
14. `tgirl.registry` has zero third-party dependencies (stdlib only)

## 5. Risk Assessment

- **Type extraction complexity:** Python's type system has many edge cases (forward refs, `from __future__ import annotations`, nested generics). The type extraction code needs thorough testing. Risk is moderate — mitigation is comprehensive test cases per type.
- **Pydantic BaseModel support:** Pydantic v2 has a different API than v1 for field inspection. Must target v2. Risk is low — Pydantic is an optional type, not a core dep.
- **Annotated range constraints:** The spec mentions `Annotated[int, Gt(0), Lt(100)]` — need to decide whether to support arbitrary annotated metadata or specific constraint types. Risk is low — start with specific types, document extension points.

## 6. Open Questions

1. **Pydantic as optional dependency:** Registry must be stdlib-only, but Pydantic BaseModel is in the type table. Should Pydantic support be conditional (try/except import) or deferred to a later feature?
2. **TypeRepresentation structure:** The spec doesn't prescribe the internal representation of extracted types. A recursive algebraic data type (sum type via dataclasses) seems natural. Confirm this approach.
3. **Annotated constraint types:** Use our own constraint types (e.g., `tgirl.types.Gt`, `tgirl.types.Lt`) or support third-party ones (annotated-types, Pydantic's)? Starting with our own keeps the stdlib-only constraint.

## 7. Out of Scope

- Grammar generation (tgirl.grammar) — depends on registry but is a separate feature
- Hy compilation (tgirl.compile) — separate feature
- Transport, sampling, bridge, serve modules — later phases
- CI/CD pipeline (GitHub Actions) — separate feature
- Any actual LLM inference or model loading
