# Plan Review: lingo-native

## Verdict: APPROVED
## Reviewer Stance: Team — Senior Training Partner + Proposer
## Date: 2026-03-16
## Mode: Agent Team (concurrent review + revision, tech lead active)

## Yield Points Found

### 1. Missing `:+` (addendum) operator in TDL parser
**Severity:** HIGH
**Evidence:** ERG uses `:+` in `delims.tdl` (38 occurrences) and `lextypes.tdl` (3 occurrences) to merge features into existing types. Parser only handled `:=`.
**Proposer Response:** Accepted. Three-pass hierarchy build (types → addenda → subsumption). `is_addendum` field on `TdlDefinition`. New tests for `:+` syntax.
**PRP Updated:** Yes — Task 1 and Task 2 revised.

### 2. `strict=True` zip crashes on 11-to-12 source transition
**Severity:** HIGH
**Evidence:** `modulation.py` lines 338 and 552 use `zip(raw_sources, cfg.conditioners, strict=True)`. When sources grow to 12 but config stays at 11, ValueError. At least 6 test assertions hardcoded to 11.
**Proposer Response:** Accepted. Lockstep change list for Task 7: conditioners, prev_smoothed, matrix_flat, and all test assertions. `__post_init__` migration logic for backward-compatible config loading.
**PRP Updated:** Yes — Task 7 revised with explicit file+line change list.

### 3. Token prefix matching is combinatorially explosive
**Severity:** Medium
**Evidence:** Single-character BPE tokens like "a" match thousands of lexicon words, producing trivially high coherence for nonsense.
**Proposer Response:** Accepted. Prefix matching dropped for v1. Exact whole-word matching only. Leading-space stripping retained for BPE normalization (` the` → `the`).
**PRP Updated:** Yes — Task 4 revised.

### 4. ERG file naming errors and incomplete include coverage
**Severity:** Medium
**Evidence:** PRP confused `letypes.tdl` (1,974 lines) with `lextypes.tdl` (29,966 lines). Planned to parse only 3 files, but ERG needs ~30 TDL files loaded via recursive `:include` from `english.tdl`.
**Proposer Response:** Accepted. File names corrected, `load_grammar()` mandated to follow includes recursively, type counts corrected.
**PRP Updated:** Yes — Task 1 and Task 3 revised.

### 5. Include resolution underspecified
**Severity:** Medium
**Evidence:** ERG includes have inconsistent extensions, subdirectory paths, section-scoped semantics, and commented-out includes. PRP said "caller resolves paths" without specifying how.
**Proposer Response:** Accepted. `resolve_include()` function specified with extension handling, relative path resolution, section context propagation. 6 new tests.
**PRP Updated:** Yes — Task 1 revised.

### 6. `%suffix` interleaved within definitions
**Severity:** Low
**Evidence:** In `inflr.tdl`, `%suffix` appears between `:=` and the definition body, not as a standalone directive. Parser would fail on all inflectional rules.
**Proposer Response:** Accepted. Split handling: standalone `%(letter-set)` as top-level directive, embedded `%suffix` as field on `TdlDefinition`. Concrete `inflr.tdl` test added.
**PRP Updated:** Yes — Task 1 revised.

### 7. `is_accepting() = False` blocks EOS tokens
**Severity:** Low (but critical correctness)
**Evidence:** `sample_mlx.py:489` masks stop tokens when `is_accepting()` returns False. A freeform linguistic grammar with `is_accepting() = False` causes infinite generation.
**Proposer Response:** Accepted (per tech lead directive). `is_accepting()` returns True with CRITICAL docstring explaining the sampling loop interaction.
**PRP Updated:** Yes — Task 6 revised.
**Tech Lead Intervention:** Verified the bug at sample_mlx.py:489, messaged proposer directly with the fix and rationale.

## What Holds Well

- Standalone module design with zero tgirl core imports
- Type-level compatibility as v1 simplification (no full unification)
- Token-to-lexeme mapping follows proven NestingDepthHook vocab scan pattern
- Coherence signal as soft constraint through modulation matrix (not hard masking)
- GrammarState protocol compliance ensures drop-in compatibility
- Real ERG file tests at every layer (not just synthetic examples)
- Clear v1/v2 boundary: coherence signal now, chart parsing later

## Summary

7 yield points, all accepted and resolved. The two HIGHs (`:+` addendum support and strict zip migration) would have caused silent type hierarchy gaps and runtime crashes respectively. YP7 (is_accepting blocking EOS) was a subtle correctness bug caught by the training partner and confirmed by the tech lead via direct codebase verification. The revised PRP accounts for the ERG's 30-year syntax complexity and has concrete, testable revisions for every finding. Ready for implementation.
