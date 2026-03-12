# Code Review: tiered-workflow

## Verdict: APPROVED
## Reviewer Stance: Code Review Partner
## Date: 2026-02-10

## PRP Compliance

### Task 1: Add tier support to `scripts/new-feature.sh`
**Status:** Implemented as specified.
While/case argument parsing loop matches the PRP pseudocode exactly. Tier validation, conditional PRD scaffolding, tier-appropriate "Next steps" output, and commit messages all match spec. The `--tier` flag can appear before or after the feature name as designed.

### Task 2: Add tier-aware artifact checking to `scripts/close-feature.sh`
**Status:** Implemented as specified.
Tier read with `cat .push-hands-tier 2>/dev/null || echo "standard"` and case-based validation. Light tier skips all artifact checks. Standard checks PRD, PRP, plan review, code review. Full additionally checks security audit. Post-squash unstaging via `git rm -f --cached` followed by `rm -f` is clean. Commit message is tier-appropriate.

### Task 3: Update `/new-feature` slash command for tier argument
**Status:** Implemented as specified.
Argument parsing instructions are clear and include examples. Dual-path branch detection covers three cases: already on the branch, branch exists on different branch, and fresh start. Existing `.push-hands-tier` takes precedence over `--tier` flag. Light tier short-circuits before PRD generation. Full tier includes security audit reminder.

### Task 4: Update `push-hands-review.yml` for tier-aware artifact checks
**Status:** Implemented as specified.
"Read workflow tier" step reads and validates file content with case fallback to "standard". PRD, PRP, and plan review checks gated on `steps.tier.outputs.tier != 'light'`. Security audit check gated on `steps.tier.outputs.tier == 'full'`. Safety-net step warns if `.push-hands-tier` would reach main. Uses `2>/dev/null` on the git diff per PRP uncertainty log item #4.

### Task 5: Update PR template with tier field
**Status:** Implemented as specified.
Tier field at top, linked artifacts with tier-context HTML comments, checklist grouped by "All tiers", "Standard and Full tiers", "Full tier only". Includes `.push-hands-tier` removal in the "All tiers" checklist.

### Task 6: Update `push-hands.md` Section 4 with tiered lifecycle
**Status:** Implemented as specified.
Section 4.1 has three-tier diagram. Section 4.1.1 "Choosing a Tier" has heuristics table with examples. Section 4.1.2 "Escalating Mid-Flight" covers all upgrade and downgrade paths. Section 7 updated with `--tier` flag and tier-aware close behavior. Section 9 replaces "Review intensity" with tier system reference. Appendix A has Tier column. Appendix B has `.push-hands-tier` entry.

### Task 7: Update `README.md` with tiered lifecycle
**Status:** Implemented as specified.
Compact three-tier diagram, Tier column in slash commands table, Customization section updated.

### Task 8: Update `CLAUDE.md` with tiered lifecycle
**Status:** Implemented as specified.
Tiered pipeline diagram, tier metadata note, `.push-hands-tier` in Conventions section.

## Issues Found

### 1. Missing trailing newline at end of `push-hands.md`
**Category:** Convention
**Severity:** Nit
**Location:** push-hands.md:731
**Details:** The diff shows the file ends without a trailing newline (`\ No newline at end of file`). POSIX convention expects text files to end with a newline. Some tools may produce warnings or unexpected behavior.
**Suggestion:** Add a trailing newline to the last line of the file.

### 2. CI warns on missing code review for light tier but `close-feature.sh` does not
**Category:** Logic
**Severity:** Minor
**Location:** `.github/workflows/push-hands-review.yml:79`, `scripts/close-feature.sh:48`
**Details:** The CI workflow checks for code review at all tiers (no tier filter on that step), so light-tier PRs will see a "Missing code review" warning. Meanwhile, `close-feature.sh` performs no artifact checks for light tier. This is intentional per PRP Task 4 ("keep for all tiers, since it's already a warning"), and the behavior is correct — CI gives a soft reminder, the script enforces hard gates. No fix needed, just noting the asymmetry is deliberate.
**Suggestion:** None — this is working as designed.

### 3. `close-feature.sh` interactive prompt blocks non-interactive use
**Category:** Logic
**Severity:** Minor
**Location:** scripts/close-feature.sh:79, scripts/close-feature.sh:110-114
**Details:** Two `read` calls require interactive input: (1) confirming missing artifacts, (2) choosing commit type. If the script is ever piped or called from automation, these will block or read unexpected input. This is an existing pattern (pre-dates this feature), not a regression. Documenting for awareness.
**Suggestion:** Not blocking. If non-interactive support is ever needed, `read` calls could be gated behind a `--yes` or `--non-interactive` flag, but that's a separate concern.

## What's Done Well

- **Argument parsing is robust.** The while/case loop in `new-feature.sh` handles `--tier` before or after the feature name, rejects unknown flags, and validates the tier value. The error messages are clear.
- **Post-squash tier file cleanup is elegant.** Using `git rm -f --cached` after `git merge --squash` but before `git commit` avoids polluting branch history with a removal commit while ensuring the file never reaches main.
- **CI workflow follows security conventions.** Branch name passed via `env:` (not `${{ }}` interpolation), slug validated before use in file paths, tier file read validated with case statement. All consistent with the Known Gotcha documented in CLAUDE.md.
- **Slash command dual-path handling is thorough.** Three cases covered (already on branch, branch exists elsewhere, fresh start), with existing `.push-hands-tier` taking precedence to prevent tier conflicts.
- **Documentation is comprehensive and consistent.** All three documentation files (push-hands.md, README.md, CLAUDE.md) tell a coherent story at different levels of detail. The escalation section (4.1.2) adds practical value beyond the original PRD scope.
- **The implementation is additive.** Standard tier behavior is completely unchanged from v1.0. Users who don't use `--tier` see no difference.

## Summary

Clean implementation that faithfully executes all 8 PRP tasks. The tiered workflow system is well-integrated across shell scripts, slash commands, CI, PR template, and documentation. No security issues, no spec mismatches, no dead code. The two minor issues found are by-design asymmetries, not defects. The one nit (missing trailing newline) is trivial.

Recommend: create PR for merge to `main`.
