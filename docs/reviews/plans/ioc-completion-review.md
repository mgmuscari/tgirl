# Plan Review: ioc-completion

## Verdict: APPROVED
## Reviewer Stance: Team — Senior Training Partner + Proposer
## Date: 2026-03-14
## Mode: Agent Team (concurrent review + revision)

## Yield Points Found

### 1. `register_from_schema()` missing quota/scope/callable
**Severity:** HIGH
**Evidence:** `register_from_schema()` at `src/tgirl/registry.py:150` accepts only `name`, `parameters`, `description`, `return_type`. No `quota`, `scope`, or `callable` parameters. Line 205 hardcodes a no-op lambda. Bridge import design depends on all three.
**Proposer Response:** Accepted. Added Task 0 as prerequisite to extend `register_from_schema` with keyword-only `quota`, `cost`, `scope`, `timeout`, `callable_fn` parameters.
**PRP Updated:** Yes — new Task 0 with dependency chain.

### 2. MCP client session lifecycle and async/sync mismatch
**Severity:** HIGH
**Evidence:** `import_mcp_tools` was async using `mcp.stdio_client()` context manager. Session dies on context exit, but wrapped callables need it alive. `compile.py` calls tools synchronously; `session.call_tool()` is async — deadlock risk under FastAPI.
**Proposer Response:** Accepted. Redesigned with `McpConnection` class using background daemon thread with its own event loop. Sync callables bridge via `asyncio.run_coroutine_threadsafe().result(timeout)`. Function changed from async to sync.
**PRP Updated:** Yes — Task 1 rewritten with lifecycle design.

### 3. Wrong line numbers and incomplete field mapping
**Severity:** MEDIUM
**Evidence:** 4/5 codebase line references wrong. Response models omitted `total_cycles`, `quotas_consumed`, `cycle_number`, `tool_invocations`.
**Proposer Response:** Accepted. Fixed all line references. Expanded response models.
**PRP Updated:** Yes.

### 4. Session construction massively underspecified
**Severity:** MEDIUM
**Evidence:** `SamplingSession.__init__` requires 10+ args including model/tokenizer bootstrapping. PRP hand-waved as a comment.
**Proposer Response:** Accepted. Added new Task 3 with `InferenceContext` dataclass and `load_inference_context(model_id, backend)` function. Explicit MLX and torch construction paths.
**PRP Updated:** Yes — new task, tasks renumbered.

### 5. `_schema_type_to_repr` missing object/null handling
**Severity:** MEDIUM
**Evidence:** `_schema_type_to_repr` at `src/tgirl/registry.py:36-61` has no handler for JSON Schema `"object"` or `"null"`. Falls through to `AnyType()`. PRD requires `object -> ModelType`.
**Proposer Response:** Accepted. Added Part B to Task 0: recursive `ModelType` construction for `"object"`, `DictType` for propertyless objects, `PrimitiveType(kind="none")` for `"null"`.
**PRP Updated:** Yes.

### 6. API name mismatch `expose_as_mcp` vs `create_mcp_server`
**Severity:** LOW-MEDIUM
**Evidence:** TGIRL.md specifies `expose_as_mcp` (single pipeline on existing server). PRP implemented `create_mcp_server` (new server, all tools). Missing reverse `TypeRepr -> JSON Schema` mapping.
**Proposer Response:** Accepted. `expose_as_mcp` now primary API. `create_mcp_server` kept as convenience. Added `_type_repr_to_schema` reverse mapping with round-trip test.
**PRP Updated:** Yes.

### 7. Test exclusions and missing request fields
**Severity:** LOW
**Evidence:** Validation gate excluded 4 test files without justification. `GenerateRequest` missing `ot_epsilon` and `base_temperature` from TGIRL.md spec.
**Proposer Response:** Accepted. Restored `test_compile.py`. Documented hardware-dependent exclusions. Added missing fields.
**PRP Updated:** Yes.

## What Holds Well

- Task decomposition is logical and now properly sequenced with explicit dependencies
- Correct reuse of `register_from_schema` and `bfcl.py:register_bfcl_tools` patterns
- Uncertainty Log is honest and was updated as questions were resolved
- Rollback plan accounts for Task 0's modifications to existing code
- Both torch and MLX backends specified with concrete construction paths
- `McpConnection` lifecycle follows established patterns (background thread + event loop)
- Round-trip type mapping test is a strong verification mechanism

## Summary

The PRP started with 2 HIGH and 3 MEDIUM structural issues that would have blocked implementation. All 7 yield points were accepted and resolved with concrete design changes. The revised PRP adds two new prerequisite tasks (registry extension, model bootstrap), resolves the async/sync lifecycle design, fixes API naming to match spec, and expands response models. The PRP is ready for execution.
