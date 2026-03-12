# PRD: Tiered Workflow System

## Status: IMPLEMENTED
## Author: agent (Proposer stance)
## Date: 2026-02-10
## Branch: feature/tiered-workflow

## 1. Problem Statement

The push hands template targets solo developers and small teams, but the default development lifecycle requires 6 stage gates and mandatory document artifacts at every stage. A solo dev adding a small feature (rename a function, fix a validation rule, add a config option) must generate a PRD, get it approved, generate a PRP, run plan review, execute, and run code review — minimum 5 slash commands across multiple agent invocations.

This is too much ceremony for small changes. The reviewer feedback puts it directly: "A tiered system (light/standard/full) might serve solo devs better."

The cost is real: developers who find the default path too heavy will either skip stages informally (losing the audit trail) or abandon the methodology entirely. Both outcomes are worse than having an explicit lightweight path that preserves the parts of the process that matter at that scale.

**Who has this problem:** Solo developers using the template for projects with a mix of small and large changes. Small teams where not every change warrants a full PRP cycle.

**Why now:** This is the #1 structural criticism from external review of the v1.0 template. Addressing it before wider adoption prevents the template from being perceived as "only for big features."

## 2. Proposed Solution

Add three explicit workflow tiers that map change size to process weight:

### Light Tier
For small, well-understood changes: bug fixes, config tweaks, single-file refactors, documentation updates.

- **No PRD.** The commit message and branch name carry sufficient context.
- **No PRP.** The change is small enough to implement directly.
- **No plan review.** There's no plan to review.
- **Code review is optional** (recommended but not gated).
- **No security audit.**
- Gate: conventional commit + lint/type check + tests pass (existing hooks handle this).
- The developer still uses a feature branch and conventional commits — git discipline is preserved.

### Standard Tier (current default)
For features that add meaningful functionality, touch multiple files, or change behavior.

- Full pipeline: PRD → PRP → plan review → execute → code review.
- Security audit remains optional (invoked explicitly).
- This is the existing workflow, unchanged.

### Full Tier
For security-sensitive features, public API changes, or anything touching auth/payments/PII.

- Everything in Standard, plus:
- Security audit is **expected** (not just optional).
- Plan review is mandatory (cannot be skipped even for "simple-seeming" features in this tier).
- The close-feature script enforces audit artifact existence.

### How tiers are selected

The tier is a human decision, not an automated classification. The developer chooses when creating the feature. Guidance is provided (size heuristics, examples), but the developer owns the call. This avoids building a classifier that gets it wrong — the cost of choosing the wrong tier is low (you can always escalate mid-flight).

### How tiers are communicated

- The README lifecycle diagram shows all three tiers.
- Each slash command's output reminds the developer what tier they're on and what comes next.
- The `close-feature.sh` script checks for required artifacts based on tier.
- The PR template checklist adapts to the tier.

## 3. Architecture Impact

### Files affected

**Documentation (content changes):**
- `push-hands.md` — Section 4 (Development Lifecycle) needs a tiered diagram and tier descriptions. Section 9 (Adapting the Template) moves "review intensity" from a customization note to a first-class concept.
- `README.md` — Lifecycle diagram and slash commands table need tier annotations.
- `CLAUDE.md` — Development Lifecycle Pipeline section needs to reflect tiers.
- `AGENTS.md` — No stance changes needed. Stances are orthogonal to tiers.

**Scripts (behavioral changes):**
- `scripts/new-feature.sh` — Accept optional `--tier light|standard|full` flag (default: standard). Record tier choice in a metadata comment in the PRD, or in a `.tier` file on the branch. For light tier, skip PRD scaffolding.
- `scripts/close-feature.sh` — Read tier metadata. Adjust artifact checks: light tier only requires code review artifact to exist (and even that as warning, not error). Full tier requires audit artifact. **Must delete `.push-hands-tier` before squash-merge** so the file never reaches `main`.

**Slash commands (behavioral changes):**
- `.claude/commands/new-feature.md` — Accept tier as argument (e.g., `/new-feature --tier light <description>`). Create `.push-hands-tier` file. For light tier, produce a minimal feature brief (not a full PRD). For standard/full, unchanged.
- `.claude/commands/review-code.md` — No change needed (already works on any branch diff).
- `.claude/commands/generate-prp.md` — No change needed (only invoked in standard/full tiers).
- `.claude/commands/review-plan.md` — No change needed (only invoked in standard/full tiers).
- `.claude/commands/execute-prp.md` — No change needed (only invoked in standard/full tiers).
- `.claude/commands/security-audit.md` — No change needed (only invoked in full tier).

**CI/GitHub (behavioral changes):**
- `.github/workflows/push-hands-review.yml` — Read tier metadata from branch. Adjust which artifacts generate warnings vs. errors.
- `.github/PULL_REQUEST_TEMPLATE.md` — Add tier field. Adjust checklist to show tier-appropriate items.

**New files:**
- None. The tier system is metadata on existing structures, not a new subsystem.

### Data model changes
- A feature branch gains a "tier" property, stored in a `.push-hands-tier` file at repo root containing a single word: `light`, `standard`, or `full`. This file exists only on feature branches — `close-feature.sh` must delete it before the squash-merge to `main`. The file is not added to `.gitignore` (it's meant to be committed on the branch for CI visibility, then removed at merge time).

### Dependency additions
- None.

## 4. Acceptance Criteria

1. Running `./scripts/new-feature.sh my-fix --tier light` creates a feature branch without scaffolding a PRD. A `.push-hands-tier` file containing `light` exists on the branch.
2. Running `./scripts/new-feature.sh my-feature` (no flag) defaults to `standard` tier. Behavior is identical to current v1.0.
3. Running `./scripts/new-feature.sh auth-rework --tier full` creates a feature branch with standard PRD scaffold. A `.push-hands-tier` file containing `full` exists on the branch.
4. `close-feature.sh` on a `light` branch succeeds with only code passing lint/tests — no artifact warnings for missing PRD, PRP, or plan review.
5. `close-feature.sh` on a `standard` branch behaves as current v1.0 (warns on missing artifacts).
6. `close-feature.sh` on a `full` branch additionally warns if security audit artifact is missing.
7. The `/new-feature` slash command respects tier and adjusts its output accordingly — light tier produces a brief commit message, not a full PRD.
8. The README lifecycle diagram shows all three tiers clearly.
9. The PR template includes a tier field and adapts its checklist.
10. The push-hands-review.yml workflow reads tier metadata and adjusts artifact checks accordingly.
11. `push-hands.md` Section 4 documents all three tiers with guidance on when to use each.

## 5. Risk Assessment

**Low risk:**
- This is additive — the standard tier IS the current behavior. Existing users who don't use `--tier` see no change.
- No new dependencies. No new file types. No new slash commands.

**Medium risk:**
- Tier metadata storage: the `.push-hands-tier` file must never reach `main`. If `close-feature.sh` is bypassed (e.g., manual merge), the file could leak. Mitigation: `close-feature.sh` deletes it before merge, and the push-hands-review.yml workflow can warn if the file is present in the PR diff targeting `main`.
- Developers choosing light tier for changes that deserve standard. Mitigation: guidance in docs with concrete examples. The cost of under-tiering is "you skipped a review that might have caught something" — recoverable. The cost of over-tiering is "you spent 20 minutes on ceremony for a one-line fix" — also recoverable but more annoying day-to-day.

**Not a risk:**
- Tier creep / proliferation. Three tiers is the cap. The PRD explicitly does not support custom tiers or sub-tiers.

## 6. Open Questions

*All resolved — see decisions below.*

### Resolved

1. **Light tier always requires a feature branch.** No direct-to-main escape hatch. Git discipline is non-negotiable at every tier.

2. **Tier metadata uses `.push-hands-tier` file, not branch naming.** The file must never reach `main` — `close-feature.sh` deletes it before the squash-merge, and it should be removed from the staging area during merge. Branch naming stays `feature/<slug>` at all tiers.

3. **Slash commands accept tier as an argument.** `/new-feature --tier light <description>` works. The slash command creates the `.push-hands-tier` file if it doesn't exist, or reads it if it does. This means developers can use either the shell script or the slash command to start a feature — both paths produce the same metadata.

## 7. Out of Scope

- **Automatic tier classification.** The developer picks the tier. No heuristics, no line-count thresholds, no file-count rules.
- **Custom tiers or sub-tiers.** Three tiers. That's it.
- **Changes to agent stances.** Stances are orthogonal to tiers — a code review partner reviews the same way regardless of tier.
- **Worked example / walkthrough.** That's a separate piece of work (also requested in reviewer feedback) and should be its own feature.
- **Token usage or cost tracking per tier.** Useful but separate concern (v1.2 roadmap).
