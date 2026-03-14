# Plan Review: bfcl-benchmark-integration

## Verdict: APPROVED
## Reviewer Stance: Team — Senior Training Partner + Proposer
## Date: 2026-03-13
## Mode: Agent Team (concurrent review + revision)

## Yield Points Found

### 1. S-expression parsing fragility — string-level parsing won't handle complex types
**Severity:** Structural
**Evidence:** The original PRP described `sexpr_to_bfcl` as string-level parsing (splitting on spaces, manual quoting). BFCL functions use strings with spaces, nested lists, booleans, None values, and dicts. String-level parsing would silently misparse `(greet "hello world")` as 3 args instead of 2.
**Proposer Response:** Accepted. Rewrote the entire `sexpr_to_bfcl` approach in Task 2 to use `hy.read_many()` AST parser. Added 4 new test cases: `test_sexpr_to_bfcl_string_with_spaces`, `test_sexpr_to_bfcl_nested_list`, `test_sexpr_to_bfcl_boolean_none`, `test_sexpr_to_bfcl_optional_params_omitted`.
**PRP Updated:** Yes — Task 2 completely rewritten with AST-based approach, Uncertainty Log item #2 marked RESOLVED.

### 2. Name sanitization in wrong layer — `_name_map` on ToolRegistry violates separation of concerns
**Severity:** Moderate
**Evidence:** The original PRP added BFCL-specific `_name_map` dict to `ToolRegistry`, coupling a general-purpose registry to a benchmark-specific naming convention. `register_from_schema()` should be agnostic about where names come from.
**Proposer Response:** Accepted. Name sanitization moved to the caller (`register_bfcl_tools()` in `tgirl.bfcl`). `register_from_schema()` accepts names as-is. Test renamed from `test_register_from_schema_dotted_name` to `test_register_from_schema_accepts_any_name`.
**PRP Updated:** Yes — Task 1 approach and tests updated.

### 3. BFCL `ast_checker` requires `MODEL_CONFIG_MAPPING` registration — not callable directly
**Severity:** Structural
**Evidence:** `ast_checker.convert_func_name()` looks up the model name in `MODEL_CONFIG_MAPPING` to decide `underscore_to_dot` behavior. Calling without registration causes `KeyError` on dotted function names. The original PRP stated the checker was "callable directly without needing `MODEL_CONFIG_MAPPING` registration" — this was incorrect.
**Proposer Response:** Accepted. Added `ModelConfig` runtime registration step to Task 3. Updated Codebase Analysis, Integration Points, and Uncertainty Log item #4 with correct information about the dependency.
**PRP Updated:** Yes — Task 3 revised, Codebase Analysis corrected.

### 4. BFCL data access via hardcoded filesystem paths
**Severity:** Moderate
**Evidence:** The original PRP referenced BFCL data at absolute filesystem paths. After pip upgrades or virtualenv changes, these paths break. Should use `importlib.resources.files()` for package-relative access.
**Proposer Response:** Accepted. Updated `load_test_data()` and `load_ground_truth()` in Task 2 to use `importlib.resources.files('bfcl_eval') / 'data'`. Updated Codebase Analysis to document the correct access pattern.
**PRP Updated:** Yes — Task 2 approach and Codebase Analysis updated.

### 5. Task 4 violated TDD mandate
**Severity:** Moderate
**Evidence:** CLAUDE.md mandates TDD for all implementation. The original Task 4 described itself as "not a TDD task" — debugging/iteration without test-first discipline. While exploratory runs are needed to discover format issues, fixes must follow RED → GREEN → REFACTOR.
**Proposer Response:** Accepted. Rewrote Task 4 to combine exploratory validation with TDD fixes: discover issues via benchmark runs, then write failing tests before fixing. Added `tests/test_bfcl.py` to Task 4 files.
**PRP Updated:** Yes — Task 4 approach rewritten.

### 6. `entry["question"]` structure underspecified
**Severity:** Minor
**Evidence:** The `question` field is `list[list[dict]]` (turns × messages), not just "nested list of messages." For single-turn, `entry["question"][0]` yields the message list for the first turn.
**Proposer Response:** Accepted. Added explicit type annotation and access pattern to Codebase Analysis and Task 3.
**PRP Updated:** Yes.

## What Holds Well

- **Overall architecture** — clean separation between `register_from_schema()` (general-purpose), `tgirl.bfcl` (adapter), and `benchmarks/run_bfcl.py` (runner)
- **Type mapping** — comprehensive coverage of all BFCL JSON schema types to tgirl's TypeRepr system
- **Irrelevance strategy** — leveraging the fact that freeform-only output naturally fails BFCL's decode_ast
- **Rollback plan** — fully additive changes, clean revert path

## Summary

All structural yield points were addressed in PRP revisions. The most important fixes were: (1) replacing string-level s-expression parsing with Hy AST parsing, (2) correcting the false assumption about `ast_checker` being callable without model registration, and (3) moving name sanitization from the registry to the BFCL adapter layer. The plan is now structurally sound and ready for implementation.
