# Code Review: initial-scaffolding

## Verdict: REQUESTS CHANGES
## Reviewer Stance: Code Review Partner
## Date: 2026-02-10

## Spec Compliance

Reviewed against `push-hands.md` (PRD v1.0), which serves as the specification for this initial template build.

### Structure (Section 3 of spec)
- **All 7 slash commands present:** `.claude/commands/{new-feature,generate-prp,review-plan,execute-prp,review-code,security-audit,update-claude-md}.md` — matches spec
- **All scripts present:** `scripts/{setup,new-feature,close-feature,worktree-setup}.sh` — matches spec (note: `setup.sh` is present in implementation but missing from spec's tree diagram — see Issue #4)
- **All hooks present:** `scripts/hooks/{pre-commit,pre-push,commit-msg}` — matches spec
- **All doc templates present:** `docs/{PRDs,PRPs,audits,decisions}/TEMPLATE.md` — matches spec
- **All GitHub config present:** Issue templates, PR template, CI workflow, push-hands review workflow — matches spec
- **Placeholder directories:** `src/.gitkeep`, `tests/.gitkeep`, `docs/reviews/{plans,code}/.gitkeep` — correct

### Content Quality
- **Slash commands:** Each correctly implements the stance described in AGENTS.md and the lifecycle described in Section 4 of the spec
- **AGENTS.md:** Faithfully represents the five stances from Section 6 of the spec
- **README.md:** Concise, correct, links to the full PRD
- **Templates:** Match the formats described in the spec
- **Scripts:** Implement the git workflows described in Section 7

## Issues Found

### 1. CLAUDE.md "Repository State" is outdated
**Category:** Convention
**Severity:** Blocking
**Location:** CLAUDE.md:37-38
**Details:** CLAUDE.md says "This repository is currently in its initial state — the PRD (`push-hands.md`) defines the complete template specification. The repository needs to be built out from this PRD into the full template structure described within it." The repository IS built out. This is the first thing Claude Code reads — incorrect state information here will mislead every future agent invocation.
**Suggestion:** Update to reflect that the template is complete. Remove or rewrite the "Repository State" section. The "Target Template Structure" section can become just "Repository Structure" since it's no longer a target — it's actual.

### 2. commit-msg hook doesn't enforce first-line-only checking
**Category:** Logic
**Severity:** Significant
**Location:** scripts/hooks/commit-msg:7
**Details:** `grep -qE "$PATTERN" "$1"` checks ALL lines of the commit message file. A commit message with an invalid first line but a body line like "docs: some note" would pass the hook. The hook should only validate the first line of the commit message.
**Suggestion:** Change to `head -1 "$1" | grep -qE "$PATTERN"` to only check the first line.

### 3. commit-msg hook regex doesn't enforce total line length
**Category:** Logic
**Severity:** Significant
**Location:** scripts/hooks/commit-msg:5
**Details:** The pattern `.{1,72}` after the type prefix only constrains the description part to 1-72 characters, but since there's no `$` anchor, `grep` will match any prefix. A description longer than 72 characters still matches because grep finds a match in the first 72 characters. The spec says "max 72 chars first line" (the entire line, not just the description). The regex allows lines well over 72 total characters.
**Suggestion:** Add `$` anchor to enforce end-of-line, and consider checking total line length separately: `[ $(head -1 "$1" | wc -c) -gt 73 ] && echo "First line exceeds 72 characters" && exit 1`

### 4. push-hands.md tree diagram missing setup.sh
**Category:** Convention
**Severity:** Minor
**Location:** push-hands.md:98-106
**Details:** Section 3's directory tree doesn't list `scripts/setup.sh`, but the script exists and is referenced in Sections 8 and 9 of the same document. CLAUDE.md correctly lists it. Internal spec inconsistency.
**Suggestion:** Add `├── setup.sh` to the scripts section of the tree diagram.

### 5. push-hands-review.yml doesn't handle non-feature branches
**Category:** Logic
**Severity:** Significant
**Location:** .github/workflows/push-hands-review.yml:18-21
**Details:** The slug extraction `SLUG="${BRANCH#feature/}"` runs on ALL PRs, not just those from `feature/*` branches. For a PR from `hotfix/urgent-fix`, the slug would be `hotfix/urgent-fix` and all artifact checks would look in wrong paths and produce spurious warnings.
**Suggestion:** Add an early check: if the branch doesn't start with `feature/`, either skip the artifact checks or provide a clear message that artifact checking only applies to feature branches.

### 6. new-feature.sh doesn't validate slug format
**Category:** Logic
**Severity:** Minor
**Location:** scripts/new-feature.sh:15
**Details:** The feature name from `$1` is used directly as a branch name and file path without validation. The spec mandates "lowercase, hyphen-separated" slugs (Appendix B). Uppercase letters, spaces, or special characters would create branches/files that violate conventions.
**Suggestion:** Add input validation: `if ! echo "$1" | grep -qE '^[a-z0-9]+(-[a-z0-9]+)*$'; then echo "Error: feature name must be lowercase, hyphen-separated"; exit 1; fi`

### 7. push-hands.md repository name doesn't match actual
**Category:** Convention
**Severity:** Minor
**Location:** push-hands.md:7
**Details:** Line 7 says `Repository: ontologi/push-hands` but the repository directory is `ontologi-push-hands`. Verify the intended GitHub repository name and update if needed.
**Suggestion:** Confirm intended repo name and update to match.

### 8. close-feature.sh hardcodes `feat:` commit type
**Category:** Logic
**Severity:** Minor
**Location:** scripts/close-feature.sh:78
**Details:** The squash-merge commit message always uses `feat: ${FEATURE_NAME}`. If the feature branch represents a fix, refactor, or other type, the commit message would be inaccurate per Conventional Commits. This is a minor concern since the primary use case is features, but it breaks convention for non-feature work that uses the feature branch pattern.
**Suggestion:** Either document that this script is specifically for feature merges, or prompt for the commit type.

## What's Done Well

- **Comprehensive coverage.** Every artifact specified in push-hands.md Section 3 exists. Nothing was skipped.
- **Slash command quality.** Each command correctly implements its stance, has clear instructions, and follows the lifecycle gates. The review-plan.md command's "read it again, looking for what's missing" instruction is particularly well-crafted.
- **AGENTS.md tone.** The stance descriptions successfully function as context primes, not just role descriptions. The distinction between "role" and "stance" in the preamble is a strong framing.
- **Script robustness.** Scripts use `set -e`, check for required arguments, provide usage examples, and give clear next-steps guidance.
- **Template completeness.** All four template files (PRD, PRP, audit, ADR) match their spec definitions and are immediately usable.
- **README conciseness.** The README gives a clear overview without duplicating the full PRD, and correctly links to push-hands.md for details.
- **Workflow design.** The push-hands-review.yml artifact checker is a clever use of GitHub Actions to enforce the process at the CI level.

## Summary

The template scaffolding is comprehensive and well-executed. The structure matches the spec, the slash commands are well-crafted, and the supporting infrastructure (scripts, hooks, workflows, templates) is solid.

Two issues need fixing before commit:
1. **CLAUDE.md state is outdated** — it describes the repo as unbuilt, which will mislead agents
2. **commit-msg hook has logic bugs** — it doesn't enforce what it claims to (first-line-only, total length)

The push-hands-review.yml non-feature branch handling is also significant but less urgent since it only manifests in CI, not local development.

Recommend fixing the Blocking and Significant issues, then proceeding to initial commit.
