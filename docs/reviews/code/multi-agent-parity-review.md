# Code Review: multi-agent-parity

## Verdict: APPROVED
## Reviewer Stance: Team — Proposer + Code Review Partner
## Date: 2026-02-11
## Mode: Agent Team (message-gated incremental review)

## PRP Compliance

| PRP Task | Commit | Status | Notes |
|----------|--------|--------|-------|
| 1. Portable stance definitions | `6f7852e` | Implemented as specified | 5 files in `prompts/stances/`, all Claude Code references removed |
| 2. Portable workflow templates | `08f0ded` | Implemented as specified | 7 files in `prompts/workflows/`, `{input}` placeholder, standardized sections |
| 3. Prompts README | `2e4f5ed` | Implemented as specified | ~60 lines, three-layer architecture, tables, usage instructions |
| 4. Claude Code adapter docs | `f34b52b` | Implemented as specified | Feature matrix, sync convention, team mode reference |
| 5. OpenCode adapter | `340fe0e` | Minor deviation (safer) | `"test -f *"` instead of `"test *"` in bash permissions; `webfetch: deny` on all non-proposer agents |
| 6. Cursor adapter | `54747aa` | Implemented as specified | 6 `.mdc` rules with correct frontmatter and activation patterns |
| 7. Generic adapter | `3601599` | Improved over spec | Security fix: ENVIRON + index/substr instead of awk gsub to prevent `&`/`\` metacharacter injection |
| 8. AGENTS.md refactor | `1eaaf5e` | Implemented as specified | Platform-neutral language, new Platform-Specific Enforcement section |
| 9. Core docs update | `97b73fd` | Implemented as specified | push-hands.md, CLAUDE.md, README.md all updated with multi-platform sections |
| 10. Scripts update | `4bcd616` | Improved over spec | Bash 3.x-compatible drift detection (no `declare -A`); platform adapter detection in setup.sh |

## Issues Found

### 1. Missing stance file cross-references in workflow templates
**Category:** Spec Compliance
**Severity:** Minor
**Location:** All 7 files in `prompts/workflows/`
**Details:** PRP specifies "reference `prompts/stances/X.md` and include a condensed version" in each workflow's Stance section. All 7 workflows include the condensed stance prime but none reference the canonical stance file (e.g., "Full stance definition: `prompts/stances/training-partner.md`").
**Resolution:** Not addressed during implementation. Non-blocking — the condensed primes are accurate and self-contained. Could be addressed in a follow-up commit.

### 2. OpenCode bash permission pattern narrower than PRP spec
**Category:** Spec Compliance
**Severity:** Minor
**Location:** `.opencode/agents/training-partner.md:9`, `.opencode/agents/code-reviewer.md:9`
**Details:** PRP mapping table specifies `"test *": allow` but implementation uses `"test -f *": allow`. This restricts review agents to only `test -f` (file existence) rather than `test -d`, `test -r`, etc.
**Resolution:** Accepted. More restrictive is safer. Review agents primarily need `test -f` for validation gates.

### 3. Security-auditor section heading inconsistency
**Category:** Convention
**Severity:** Nit
**Location:** `prompts/stances/security-auditor.md`
**Details:** PRP template specifies `## Key Behaviors` as the third section for all stances. Security-auditor uses `## Vulnerability Categories` instead (training-partner and skeptical-client were correctly renamed from source headings).
**Resolution:** No action needed. "Vulnerability Categories" is more descriptive than the generic "Key Behaviors" for this stance.

### 4. Extra webfetch: deny on non-proposer OpenCode agents
**Category:** Spec Compliance
**Severity:** Nit
**Location:** `.opencode/agents/{training-partner,code-reviewer,security-auditor}.md`
**Details:** `webfetch: deny` added to all 4 non-proposer agents. PRP mapping table only explicitly lists it for skeptical-client (though the PRP YAML example for training-partner also includes it).
**Resolution:** Conservative safety choice. Accepted.

### 5. Drift-detection script UX
**Category:** Convention
**Severity:** Nit
**Location:** `scripts/check-prompt-sync.sh`
**Details:** Script prints a `✓` line per workflow pair even after emitting `⚠ Drift` warnings for the same pair. This matches the PRP spec exactly but produces slightly confusing output (success checkmark alongside drift warning).
**Resolution:** Spec-compliant. No action needed.

## What's Done Well

- **Security fix in run-stage.sh (Task 7).** The proposer identified and fixed a metacharacter injection vulnerability in the PRP's specified awk command. The PRP used `awk -v input="$ARGS" '{gsub(...)}'` where `&` means "matched text" and `\` acts as an escape in gsub's replacement string. The implementation uses ENVIRON + index/substr for fully literal replacement. The code-reviewer independently verified the fix with pathological input containing `|`, `&`, `\`.

- **Bash 3.x compatibility (Task 10).** The PRP's `check-prompt-sync.sh` used `declare -A` (associative arrays) requiring bash 4+. macOS ships bash 3.2. The implementation uses colon-separated pair lists with `${var%%:*}` and `${var##*:}` parameter expansion — works everywhere.

- **Comprehensive validation suite.** The proposer ran 13 validation gates covering file existence, section structure, no Claude Code primitives leaked, no `$ARGUMENTS` in portable layer, all existing commands preserved, all doc sections present, and drift script validity.

- **Faithful platform research.** OpenCode agent definitions use the correct `permission:` field (not `tools:`), Cursor rules use `.mdc` format with proper activation patterns, and the AGENTS.md context is correctly treated as a plain markdown document (not agent definitions).

- **Clean atomic commits.** All 10 commits follow Conventional Commits, are under 72 characters, and each corresponds to exactly one PRP task with no cross-task changes.

- **Additive-only approach.** Zero changes to existing `.claude/commands/` or `.claude/agents/` files. All existing Claude Code functionality preserved. Rollback is trivial: delete new directories, revert modified files.

## Summary

All 10 PRP tasks were implemented across 10 atomic commits. The code-reviewer approved every commit with zero blocking or significant findings. The 2 minor findings (missing stance cross-references in workflows, narrower OpenCode bash permissions) are non-functional and don't affect the feature's correctness or usability.

The implementation includes two notable improvements over the PRP specification:
1. A security fix to `run-stage.sh` that prevents metacharacter injection in awk's replacement string
2. Bash 3.x compatibility in the drift-detection script for macOS users

The three-layer architecture (shared infrastructure → portable prompt templates → platform adapters) is cleanly implemented. Portable templates are genuinely platform-neutral. Adapters correctly use each platform's native format (Claude Code commands, OpenCode agents with permissions, Cursor .mdc rules, generic shell wrapper).

**APPROVED** — ready for security audit or PR creation.
