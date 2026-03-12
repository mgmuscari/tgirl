# Plan Review: grammar

## Verdict: APPROVED
## Reviewer Stance: Team — Senior Training Partner + Proposer
## Date: 2026-03-12
## Mode: Agent Team (concurrent review + revision)

## Yield Points Found

### 1. Task 2 dispatches on wrong discriminator values
**Severity:** Moderate
**Evidence:** `src/tgirl/types.py:24-30` — `PrimitiveType` has `type_tag = "primitive"` with a separate `kind` field. PRP listed `"str"`, `"int"` etc. as `type_tag` values, which would never match.
**Proposer Response:** Accepted. Restructured to two-level dispatch — first on `type_tag`, then sub-dispatch on `PrimitiveType.kind`.
**PRP Updated:** Yes (Task 2, lines 72-87)

### 2. Quota enforcement deferred without justification + interface gap
**Severity:** Structural (HIGH)
**Evidence:** TGIRL.md section 4.4 calls grammar-level quota enforcement "the critical innovation." PRP silently picked Fallback 2 without justification. `GrammarOutput` lacked quota fields despite claiming to carry quota info.
**Proposer Response:** Accepted (both parts). Added `tool_quotas` and `cost_remaining` to `GrammarOutput`. Rewrote Uncertainty Log with explicit justification: Outlines CFGGuide has no mutable state API, expanded states cause combinatorial explosion, Fallback 2 is correct (spec mandates outcome not mechanism).
**PRP Updated:** Yes (Task 1, Task 6, Uncertainty Log item 2)

### 3. Overlapping threading rules and LALR(1) ambiguity risk
**Severity:** Moderate
**Evidence:** PRP lines 126-127 defined `pipeline` with `->` in base template; lines 155-161 defined `threading` with `->` in composition template. Duplicate productions + unanalyzed mutual recursion.
**Proposer Response:** Accepted. Unified threading into composition only. Added LALR(1) safety explanation — distinct keyword prefixes after opening paren provide unambiguous lookahead.
**PRP Updated:** Yes (Task 4, Task 5)

### 4. ModelType grammar uses incorrect Hy syntax
**Severity:** Moderate
**Evidence:** PRP line 84 — `":field_name" value` syntax was wrong for both Hy dicts and keyword args.
**Proposer Response:** Partially accepted. Fixed Hy dict syntax for both DictType and ModelType. Rejected keyword-arg alternative — keyword args are function call syntax, ModelType represents inline value literals. Added Uncertainty Log item for grammar-compile interface risk.
**PRP Updated:** Yes (Task 2 DictType/ModelType, new Uncertainty Log item 6)

### 5. No Lark parse validation in test plan
**Severity:** Minor
**Evidence:** PRP line 144 claimed "valid Lark EBNF" but no test actually parsed with Lark.
**Proposer Response:** Accepted. Added `lark.Lark(grammar_text, parser="lalr")` validation to Task 4 and Task 8 tests. Added `lark>=1.0` as grammar dependency.
**PRP Updated:** Yes (Task 4 tests, Task 8 tests, codebase analysis)

### 6. Positional parameter convention can't skip optional defaults
**Severity:** Minor
**Evidence:** PRP lines 103-109 — positional ordering means supplying a later optional param requires emitting all earlier defaults.
**Proposer Response:** Partially accepted. Documented explicit "positional, trailing-optional chain" convention with justification. Rejected keyword args — preventing duplicate kwargs is not CFG-expressible without state (same class of problem as quota enforcement).
**PRP Updated:** Yes (Task 3, lines 104-108)

## What Holds Well

- Well-sequenced 8-task decomposition with clean dependency progression
- Template-based architecture is the right abstraction for future format flexibility (GBNF, regex)
- Determinism guarantee well-specified with sorted collections and hash-based verification
- Uncertainty Log is honest, well-calibrated, and was strengthened through review
- Integration test plan covers important cross-module boundaries (registry -> snapshot -> grammar -> diff)
- Consistent principled basis for design decisions: grammar stays stateless, state-dependent enforcement deferred to sampler

## Summary

The PRP had one structural issue (HIGH — quota enforcement justification and interface gap) and five moderate/minor issues. All were resolved through the review exchange: 4 fully accepted, 2 partially accepted with well-justified rejections. The two rejected sub-proposals (keyword args for ModelType, keyword args for tool parameters) were rejected on the same principled basis — they require stateful enforcement not expressible in a CFG.

The PRP is ready for implementation. **APPROVED.**
