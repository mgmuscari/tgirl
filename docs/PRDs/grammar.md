# PRD: tgirl.grammar — Dynamic CFG Generation

## Status: DRAFT
## Author: agent (Proposer)
## Date: 2026-03-12
## Branch: feature/grammar

## 1. Problem Statement

tgirl's core value proposition is that grammar-constrained generation produces valid tool call pipelines at the token level — invalid calls are inexpressible, not caught at runtime. The `tgirl.registry` module (now complete) captures tool definitions and type representations, but nothing yet converts that information into an actual context-free grammar that can constrain model output.

Without `tgirl.grammar`, the registry is metadata with no enforcement mechanism. This module is the bridge between "knowing what tools exist" and "constraining generation to only produce valid invocations of those tools."

## 2. Proposed Solution

Implement `tgirl.grammar` as a Jinja2 template-based CFG generator that consumes `RegistrySnapshot` objects and produces grammar specifications compatible with Outlines' constrained generation.

The module:

1. **Converts TypeRepr variants to CFG productions** — each type in the type system (primitives, lists, dicts, models, unions, optionals, literals, enums, annotated constraints) gets a grammar rule that constrains token output to valid values of that type.

2. **Generates tool-specific grammar rules** — each registered tool becomes a production with its name and per-parameter type productions, respecting required/optional parameter semantics.

3. **Composes full grammars from Jinja2 templates** — four template files (`base.cfg.j2`, `tools.cfg.j2`, `types.cfg.j2`, `composition.cfg.j2`) are rendered with snapshot data and composed into a complete CFG.

4. **Supports grammar diffing** — deterministic output (same snapshot → same grammar) enables structured diffing for debugging and auditing.

5. **Quota enforcement strategy** — initial implementation uses the grammar-level approach where quotas affect available tool productions. The spec allows multiple strategies; the PRP will determine which is feasible for v1.0.

## 3. Architecture Impact

### Files created
- `src/tgirl/grammar.py` — module implementation (generate, diff, types for grammar output)
- `src/tgirl/templates/base.cfg.j2` — top-level grammar structure
- `src/tgirl/templates/tools.cfg.j2` — per-tool productions
- `src/tgirl/templates/types.cfg.j2` — type-to-production mappings
- `src/tgirl/templates/composition.cfg.j2` — composition operator productions
- `tests/test_grammar.py` — unit tests
- `tests/test_integration_grammar.py` — integration tests (registry → grammar pipeline)

### Files modified
- `src/tgirl/__init__.py` — export grammar module public API

### Dependencies
- `jinja2>=3.0` (already declared as optional dep in `pyproject.toml` under `[grammar]`)

### Data model
- New types in `grammar.py` for grammar output representation (e.g., `GrammarOutput`, `GrammarDiff`, `Production`)
- Input: `RegistrySnapshot` from `tgirl.types` (no changes needed)

## 4. Acceptance Criteria

1. `tgirl.grammar.generate(snapshot)` produces a valid CFG string from any well-formed `RegistrySnapshot`.
2. Each TypeRepr variant (`PrimitiveType`, `ListType`, `DictType`, `LiteralType`, `EnumType`, `OptionalType`, `UnionType`, `ModelType`, `AnnotatedType`, `AnyType`) has a corresponding CFG production.
3. Per-tool productions correctly encode parameter names, types, required/optional semantics, and default values.
4. Composition operators (`->` threading, `let`, `if`, `try/catch`, `pmap`) are expressible in the grammar.
5. The `insufficient-resources` terminal is always available as an alternative to tool calls.
6. Grammar output is deterministic: same `RegistrySnapshot` (modulo timestamp) produces identical grammar text.
7. `tgirl.grammar.diff(grammar_a, grammar_b)` returns a structured diff of changed productions.
8. Grammar templates are separate `.j2` files under `src/tgirl/templates/`, composed by inclusion.
9. Annotated types with numeric range constraints use enumeration for ranges ≤ 256 values and constrained numeric production for larger ranges (threshold configurable).
10. All tests pass under `pytest tests/ -v` with `tgirl[grammar]` installed.

## 5. Risk Assessment

- **Outlines grammar format compatibility**: The grammar must produce output that Outlines can actually consume. Outlines uses EBNF-like syntax. Need to verify exact format requirements during implementation.
- **Quota enforcement complexity**: Stateful grammar (preferred per spec) may not be directly expressible in Outlines' CFG format. Fallback approaches (expanded states, post-grammar logit mask) add complexity or move enforcement out of the grammar.
- **Recursive type productions**: Deeply nested types (list of dicts of models with optional fields) could produce large grammars. Need to ensure template rendering handles recursion cleanly.
- **Jinja2 template debugging**: Template errors in `.j2` files can be hard to diagnose. Need good test coverage of template rendering.

## 6. Open Questions

1. What is the exact EBNF syntax Outlines expects? Need to verify against Outlines source/docs during PRP.
2. For quota enforcement in v1.0, should we implement the stateful grammar approach, the expanded-states approach, or defer to sampling loop enforcement? (Spec says outcome matters, not approach.)
3. Should `AnyType` grammar production be a permissive catch-all or should it be rejected at grammar generation time?

## 7. Out of Scope

- GBNF output format (v1.1 planned)
- Regex output format for vLLM (v1.1 planned)
- JSON Schema diagnostic output (nice-to-have, not v1.0)
- Integration with `tgirl.sample` (that module consumes grammar output but is a separate feature)
- Runtime quota tracking (that's `tgirl.sample`'s responsibility if grammar-level enforcement proves infeasible)
