# Plan Review: transport-reachable-set

## Verdict: APPROVED
## Reviewer Stance: Team — Senior Training Partner + Proposer
## Date: 2026-03-15
## Mode: Agent Team (concurrent review + revision)

## Yield Points Found

### 1. Bypass checks break under projection
**Severity:** HIGH
**Evidence:** `valid_ratio_threshold=0.5` fires incorrectly when denominator shrinks from 248k to ~8k. 5k valid tokens in 8k reachable = 0.625 ratio → OT bypassed, defeating the optimization.
**Proposer Response:** Accepted. Bypass checks now run on full-vocab tensors before projection. Source mass computed from original softmax.
**PRP Updated:** Yes — Task 3/4 revised.

### 2. ESCAPED_STRING inflates reachable set to ~168k tokens
**Severity:** HIGH
**Evidence:** 68% of Qwen3.5's 248k vocabulary contains printable ASCII. Any grammar with a `str` parameter includes `ESCAPED_STRING`, making the reachable set ~168k — not the assumed 5-10k. Validated empirically: 168,708 tokens match.
**Proposer Response:** Partially accepted. Sizing claims revised. Task 6 now requires two benchmark profiles (numeric-only vs string-inclusive). Task 7 (NNMF) upgraded from optional to conditional (mandatory if |R| > 150k). Acceptance criterion #4 qualified per grammar state type.
**PRP Updated:** Yes.

### 3. Unreachable token mass discarded, not transported
**Severity:** HIGH
**Evidence:** Softmax over projected logits renormalizes away mass from unreachable tokens. Full-vocab path would transport that mass by embedding similarity; projected path distributes it via renormalization.
**Proposer Response:** Partially accepted. Trade-off documented as acceptable — unreachable token embeddings are semantically distant noise, and the bypass path (which fires on 80% of tokens currently) already does implicit mass discarding. The projected path is strictly better than bypass.
**PRP Updated:** Yes — trade-off documented explicitly.

### 4. GrammarOutput determinism invariant
**Severity:** Medium (downgraded from HIGH)
**Evidence:** `reachable_tokens` depends on tokenizer, not just registry snapshot. Same snapshot → different GrammarOutput with different tokenizers. CLAUDE.md says "Same registry snapshot must produce same grammar."
**Proposer Response:** Accepted in spirit. `reachable_tokens` is transport metadata, not grammar structure. `diff()` operates on productions, `snapshot_hash` excludes reachable tokens. Tests added to verify invariant preservation.
**PRP Updated:** Yes.

### 5. Regex terminal extraction is fragile
**Severity:** Medium
**Evidence:** PRP hardcodes known terminals but grammar generates many dynamically (dict/list delimiters, composition keywords, enumerated integers). Lark's own `lark.Lark(grammar_text).terminals` API would be authoritative.
**Proposer Response:** Accepted. Task 1 rewritten to use Lark parser API instead of regex heuristics. Four new test cases for dynamic terminals.
**PRP Updated:** Yes.

## What Holds Well

- The core optimization concept is architecturally sound — restricting OT's domain is the right approach
- Zero-coupling invariant preserved: `compute_reachable_set` in grammar.py, reachable set passed as plain `frozenset[int]` to transport
- The bypass-on-full-vocab / project-then-transport split (post-YP1 revision) is clean
- Task decomposition follows the same pattern as previous PRPs
- Empirical validation built into Task 6 (two benchmark profiles)
- Conditional NNMF (Task 7) avoids premature optimization while acknowledging the string-parameter reality

## Summary

The PRP started with 3 HIGH findings that would have produced incorrect behavior (false OT bypasses), inflated performance claims (168k reachable set vs claimed 5-10k), and silent semantic changes (mass discarding). All were addressed with concrete revisions. The key architectural insight from the review: the optimization is grammar-state-dependent — massive for numeric terminals (~1k reachable, OT always runs), marginal for string terminals (~168k reachable, NNMF needed). The PRP now reflects this reality.
