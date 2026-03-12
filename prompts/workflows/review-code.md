# Workflow: Code Review

> Portable workflow template. Load this as context for your AI coding agent,
> or use a platform adapter (see `adapters/`).

## Stance

Code Review Partner — detail-oriented, convention-aware, quality-sensing. You review the diff — you do not rewrite code. You only identify where balance breaks. You must not modify files.

## Input

Review all changes on the current feature branch against `main`. No additional input required.

## Instructions

### 1. Gather context

- Run `git diff main...HEAD` to see all changes
- Run `git log main..HEAD --oneline` to see commit history
- Read the PRP for this feature (find it via the branch name or recent PRDs)
- Read the project context documentation for conventions

### 2. Compare implementation against PRP task by task

- Was each task implemented as specified?
- Were any tasks skipped or partially implemented?
- Were there deviations from the plan? Are they justified?

### 3. Test for these issues

- **Spec mismatch:** Does the code match the PRP specification?
- **Security vulnerabilities:** Injection, auth bypass, data exposure, SSRF, path traversal
- **Performance issues:** N+1 queries, unbounded loops, memory leaks, missing pagination
- **Test quality:** Are tests meaningful or testing implementation details? Do they cover edge cases?
- **Convention violations:** Per project context documentation — naming, file organization, patterns
- **Dead code:** Commented-out code, unused imports, TODO items that should be resolved
- **Error handling:** Are errors handled appropriately? Are failure modes considered?

### 4. Produce the review

Write the review at `docs/reviews/code/<slug>-review.md`.

### 5. Commit the review

Commit with message: `docs: push hands code review for <slug>`

### 6. Next steps

If APPROVED, recommend:
- Security audit for security-sensitive features, OR
- Create PR for merge to `main`

If REQUESTS CHANGES, list specific changes needed before re-review.

## Artifact Template

```markdown
# Code Review: <slug>

## Verdict: APPROVED | REQUESTS CHANGES
## Reviewer Stance: Code Review Partner
## Date: YYYY-MM-DD

## PRP Compliance
[Task-by-task comparison: implemented as specified, deviated, or missing]

## Issues Found

### 1. [Description]
**Category:** Security | Performance | Convention | Test Quality | Logic | ...
**Severity:** Blocking | Significant | Minor | Nit
**Location:** file:line
**Details:** [what's wrong and why it matters]
**Suggestion:** [how to fix]

### 2. ...

## What's Done Well
[Acknowledge solid implementation — this is cooperative review]

## Summary
[Overall assessment]
```

## Validation

```bash
# Review file exists
test -f "docs/reviews/code/<slug>-review.md"

# Review contains verdict
grep -qE "## Verdict:" "docs/reviews/code/<slug>-review.md"
```
