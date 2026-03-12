# Plan Review: multi-agent-parity

## Verdict: APPROVED
## Reviewer Stance: Team — Senior Training Partner + Proposer
## Date: 2026-02-11
## Mode: Agent Team (concurrent review + revision)

## Yield Points Found

### 1. PRD AC #3 Cannot Be Satisfied As Written — Commands Can't Delegate
**Severity:** Structural
**Evidence:** PRD Acceptance Criterion #3 specifies "Claude Code `.claude/commands/*.md` files are refactored to delegate to `prompts/workflows/`." However, Claude Code commands are self-contained markdown files loaded as-is — they have no import/include mechanism. The original PRP acknowledged this constraint (Section 2, "Key Constraint") but did not explicitly address the PRD deviation or provide an alternative for the PRD's HIGH-risk "Lint/CI check" mitigation.
**Proposer Response:** Accepted and revised. Added explicit PRD AC #3 deviation documentation (PRP lines 106-110) explaining the trade-off: reliability of self-contained commands over DRY purity. Created `scripts/check-prompt-sync.sh` (Task 10 expanded) as the drift-detection mechanism, addressing the PRD's risk mitigation intent. Added Uncertainty Log item #7 documenting this as a deliberate deviation.
**PRP Updated:** Yes — Section 2 expanded, Task 10 split into Part A (sync script) + Part B (setup.sh), Uncertainty Log items #7-8 added.

### 2. Shell Injection in run-stage.sh via sed Metacharacters
**Severity:** Structural
**Evidence:** Task 7's `run-stage.sh` used `sed "s|{input}|${ARGS}|g"` for argument substitution. If `ARGS` contains `|`, `&`, `\`, or other sed-special characters (common in file paths and descriptions), the substitution corrupts or injects into the output. This is the exact class of vulnerability flagged in CLAUDE.md's Known Gotchas (script injection vectors).
**Proposer Response:** Accepted and revised. Replaced `sed` with `awk -v input="$ARGS" '{gsub(/\{input\}/, input)}1'` which treats the replacement as a literal string. Added validation test for pathological inputs containing `|`, `&`, `\`.
**PRP Updated:** Yes — Task 7 run-stage.sh implementation and validation updated.

### 3. Validation Tests Check Existence Only, Not Content Quality
**Severity:** Moderate
**Evidence:** Task 1 validation only checked `test -f` (file exists) and absence of forbidden terms. A file containing only a header and no substance would pass. Task 2 similarly lacked section structure validation. For a documentation-only feature, content quality IS the deliverable — existence-only checks provide false confidence.
**Proposer Response:** Accepted and revised. Task 1 validation now checks that each stance file contains `## Character` and `## Constraints` sections. Task 2 validation now verifies that each workflow contains all four required template sections (`## Stance`, `## Input`, `## Instructions`, `## Validation`).
**PRP Updated:** Yes — Task 1 and Task 2 validation sections expanded with section-presence checks.

### 4. update-project-docs Workflow Underspecified
**Severity:** Moderate
**Evidence:** Task 2's `update-project-docs.md` was described in one paragraph. Unlike other workflows that had detailed notes on stance, structure, and edge cases, this one only said "Generalized from `/update-claude-md`. Updates whatever the project's primary context document is." It lacked: how the target file is determined, what happens when multiple context files exist, and how the platform-specific adapters should handle target selection.
**Proposer Response:** Accepted and revised. Expanded the `update-project-docs.md` description with: two inputs (`{input}` for lesson, `{target}` for target file), a target detection heuristic (CLAUDE.md → AGENTS.md → .cursorrules → .windsurfrules → ask user), and a note that platform-specific adapters should hardcode their target.
**PRP Updated:** Yes — Task 2 update-project-docs description significantly expanded.

### 5. OpenCode Permission Mappings Are Untested Best-Effort
**Severity:** Moderate
**Evidence:** Task 5's permission mappings (e.g., `"git *": allow`, `"test -f *": allow`) were derived from documentation research, not tested against a running OpenCode instance. The PRP's Uncertainty Log item #1 rated this 7/10 confidence but the task description presented the mappings as definitive. The inconsistency between the training-partner/code-reviewer mappings (defaulting to `ask`) vs the YAML example (defaulting to `deny`) further suggested the permissions weren't fully thought through.
**Proposer Response:** Accepted and revised. Added "Permissions caveat" to Task 5 README.md description noting these are best-effort mappings. Added "Design note" explaining the `ask` vs `deny` default rationale: `ask` for review agents (to allow legitimate unexpected commands), `deny` for skeptical-client (no shell access). Fixed the YAML example to use `"*": ask` consistent with the mapping table.
**PRP Updated:** Yes — Task 5 approach expanded with caveat and design note.

### 6. PRD Architecture Divergences Not Documented
**Severity:** Minor
**Evidence:** The PRP diverged from the PRD's architecture sketch in three ways: (a) PRD shows `.agents/opencode/` but PRP uses `.opencode/agents/` (the actual OpenCode convention), (b) PRD shows flat `prompts/` layout but PRP uses subdirectories, (c) PRD AC #1 says "six stages" but PRP creates 7. These are improvements but were undocumented, creating a gap between PRD and PRP that could confuse future reviewers.
**Proposer Response:** Accepted and revised. Added Uncertainty Log item #8 explicitly documenting all three divergences with rationale (research findings, clarity, PRD AC #12 requirement).
**PRP Updated:** Yes — Uncertainty Log item #8 added.

## What Holds Well

- **Three-layer architecture is sound.** The separation of shared infrastructure, portable templates, and platform adapters is a clean decomposition that matches how the target platforms actually work.
- **Research depth is exceptional.** The codebase analysis (Section 2) and external research (Section 3) ground every decision in actual platform documentation, not assumptions. The AGENTS.md finding (context document, not agent definitions) fundamentally shaped the approach.
- **Task decomposition is appropriately granular.** 10 tasks, each independently committable and testable. The dependency ordering is correct (stances → workflows → adapters → docs → setup).
- **Additive-only approach is safe.** Existing Claude Code functionality remains completely untouched. Rollback is trivial (delete new directories, revert modified files).
- **Uncertainty log is honest.** All 8 items acknowledge real unknowns with calibrated confidence scores. The PRD deviation documentation (items #7-8) is particularly valuable.

## Summary

The PRP underwent substantive revision during team review. Six yield points were raised — two structural, three moderate, one minor — and all were accepted by the proposer. The most significant changes were:

1. Addition of `scripts/check-prompt-sync.sh` to address drift risk (structural)
2. Fixing the shell injection vulnerability in `run-stage.sh` (structural)
3. Strengthening validation tests from existence-only to content-aware
4. Explicitly documenting PRD deviations with rationale

The PRP is now a well-grounded implementation plan with honest uncertainty tracking, sound architecture, and practical validation. All structural yield points have been addressed in the revision.

**APPROVED** — ready for execution.
