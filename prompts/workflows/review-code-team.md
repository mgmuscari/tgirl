# Workflow: Code Review (Team)

> Portable workflow template. Dual-agent code review — a reviewer surfaces
> findings and a defender responds. Load this for any platform; for Claude
> Code use the `/review-code-team` command which wires this up with agent
> teams.

## Stance

Two concurrent stances:
- **Code Review Partner** — detail-oriented, convention-aware. Does not modify code.
- **Implementation Defender** — evidence-based, responds with defended/acknowledged/fixed. Write access bounded to finding scope (see `prompts/stances/defender.md`).

## Input

A feature branch with a PRP at `docs/PRPs/<slug>.md` and implementation commits to review. If no PRP exists for this branch, **refuse** and **redirect** the user to the sequential code-review workflow (`/review-code` in Claude Code, or the single-agent workflow on other platforms). Do not start the team exchange.

## Instructions

### 1. Gather context

- Derive the feature slug from the branch name (strip `feature/` prefix).
- If no PRP exists at `docs/PRPs/<slug>.md`, refuse and redirect to the sequential code-review workflow.
- Otherwise:
  - Read the PRP at `docs/PRPs/<slug>.md`.
  - Read the plan review at `docs/reviews/plans/<slug>-review.md` if present.
  - Read project context documentation for conventions.
  - Run `git diff main...HEAD` to capture the full diff.

### 2. Reviewer: surface findings one at a time

The reviewer compares the diff to the PRP task by task, then does a sweep for spec mismatch, security, performance, test quality, convention violations, dead code, and error handling. Each finding includes:

- Category
- Severity (Blocking | Significant | Minor | Nit)
- Location (file:line)
- Details
- Suggestion

The reviewer does not modify any files — findings are surfaced only.

### 3. Defender: respond to each finding

For each finding the defender replies with one of three response modes:

- **Defended** — the finding is invalid; cite a PRP task, a plan-review yield point, or a convention rule as evidence.
- **Acknowledged** — the finding is valid but non-blocking; note it for the record.
- **Fixed** — the finding is valid and Blocking; edit the code (within the defender's allowed edit scope — see `prompts/stances/defender.md`), re-run the test command, and commit atomically. Include the commit SHA in the response.

The defender must not weaken or delete tests to resolve a finding. The defender must not touch forbidden-scope paths (the PRP, project context docs, methodology files, or files outside the PR diff scope).

### 4. Convergence

Each finding closes after one reviewer → defender exchange. A second round is allowed only if the defender's fix introduces a new issue the reviewer catches; after the second round, remaining disagreements are documented in the artifact as open items.

### 5. Produce the review

Write the review at `docs/reviews/code/<slug>-review.md` using the artifact template below.

### 6. Commit

`docs: dialectic code review for <slug> (team mode)` — or the equivalent convention for your platform.

## Artifact Template

```markdown
# Code Review: <slug>

## Verdict: APPROVED | REQUESTS CHANGES
## Reviewer Stance: Team — Code Reviewer + Implementation Defender
## Date: YYYY-MM-DD
## Mode: <team | sequential adapter>

## PRP Compliance
[Task-by-task: implemented as specified, deviated, or missing]

## Issues Found

### 1. [Description]
**Category:** Security | Performance | Convention | Test Quality | Logic | ...
**Severity:** [initial] → [final after exchange]
**Location:** file:line
**Details:** [what was found]
**Defender Response:** defended (evidence) | acknowledged | fixed (commit SHA)
**Resolution:** resolved | open | downgraded

### 2. ...

## What's Done Well
[Strengths surfaced during the exchange — cooperative, not adversarial]

## Summary
[Overall assessment. Finding counts by final severity. Whether all Blocking issues were resolved.]
```

## Validation

```bash
# Review artifact was written at the expected path
test -f "docs/reviews/code/<slug>-review.md"

# Required sections are present
grep -qE "^## Verdict:" "docs/reviews/code/<slug>-review.md"
grep -qE "Defender Response" "docs/reviews/code/<slug>-review.md"
```
