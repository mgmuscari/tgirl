You are orchestrating a **dual-agent Dialectic code review** using agent teams. You are the team lead.

The Code Reviewer examines the diff against the PRP specification while the Implementation Defender explains decisions and fixes blocking issues. Findings flow via messages as they're discovered — the defender responds to each before the reviewer moves on.

## Instructions

### 1. Gather context

- Derive the feature slug from the current branch name (strip `feature/` prefix).
- If `docs/PRPs/{slug}.md` does not exist, print exactly:
  "/review-code-team requires a PRP at docs/PRPs/{slug}.md. Use /review-code instead, or run /new-feature first to generate a PRP."
  and exit without creating a team, spawning agents, or writing any artifact. Do NOT call `TeamCreate`.
- Otherwise proceed:
  - Run `git diff main...HEAD --stat` to see scope of changes.
  - Run `git log main..HEAD --oneline` to see commit history.
  - Read the PRP at `docs/PRPs/{slug}.md`.
  - Read the plan review at `docs/reviews/plans/{slug}-review.md` if it exists.
  - Read CLAUDE.md for project conventions.

### 2. Create the team

- Call `TeamCreate` with `team_name: "review-{slug}"`.

### 3. Create tasks (no dependencies — coordination via messages)

Create two tasks via `TaskCreate`:

**Task A: "Review implementation against PRP specification"**
Description: You are the Code Reviewer. Read the PRP at `docs/PRPs/{slug}.md`, the plan review at `docs/reviews/plans/{slug}-review.md` (if exists), and CLAUDE.md. Run `git diff main...HEAD` to see all changes. For each PRP task: compare the implementation against the specification. Then do a full-diff sweep for: spec mismatches, security vulnerabilities, performance issues, test quality (meaningful assertions, edge cases, not testing implementation details), convention violations (per CLAUDE.md), dead code, and error handling gaps. Send each finding as a structured message to "defender" with category, severity, location, details, and suggestion. Severity levels: Blocking (must fix before merge), Significant (should fix), Minor (nice to fix), Nit (style only). **WAIT for defender's response before sending the next finding** — the exchange on each finding must close before moving on. After all findings and responses are complete, send a final summary to the team lead. **Final summary must be self-contained:** for each finding list category, severity (initial and final after defender response), location, details, defender's response, and whether the issue was resolved/acknowledged/disputed.

**Task B: "Defend implementation decisions and fix blocking issues"**
Description: You are the Implementation Defender. Read the PRP at `docs/PRPs/{slug}.md`, the plan review at `docs/reviews/plans/{slug}-review.md` (if exists), and CLAUDE.md. Run `git diff main...HEAD` to understand all changes. Wait for findings from "code-reviewer". For each finding: if the implementation decision was intentional, explain why with evidence (cite PRP task, plan review yield point, or CLAUDE.md convention). If the finding is valid and Blocking, fix the code and run tests, then report what you changed. If valid and non-Blocking, acknowledge it. Send your response to "code-reviewer" — they are waiting. After all findings are resolved, send a final summary to the team lead. **Final summary must be self-contained:** for each finding list your response (defended/acknowledged/fixed), what was changed if anything, and the commit SHA if a fix was made.

**Convergence rule:** Each finding follows: reviewer reports → defender responds → finding is CLOSED. No further exchange on a closed finding. The review is complete when: (a) the reviewer has sent all findings AND a summary to the team lead, AND (b) every finding has been responded to and the defender has sent a summary. Maximum 2 rounds per finding (reviewer raises → defender responds; if defender's fix introduces a new issue, reviewer may flag it once more).

**Do NOT use `addBlockedBy`.** Both agents start simultaneously. The Defender reads the PRP and diff while the Reviewer begins analysis.

### 4. Spawn teammates (both in parallel)

Spawn both teammates using the Agent tool with `run_in_background: true` and `model: "opus"`:

- `name: "code-reviewer"`, `subagent_type: "code-reviewer"`, `team_name: "review-{slug}"`, `model: "opus"`

  Prompt: "**Your task:** Review the implementation on this branch against the PRP at docs/PRPs/{slug}.md. Read the PRP, plan review (if exists), and CLAUDE.md. Run `git diff main...HEAD` for the full diff and use `git show {sha}` for individual commits. Review all changes. For each finding: Category, Severity (Blocking | Significant | Minor | Nit), Location (file:line), Details, Suggestion. For convention findings, cite the specific CLAUDE.md rule. Send each finding as a structured message to 'defender'. WAIT for defender's response before moving to the next finding. Send final summary to team lead when done."

- `name: "defender"`, `subagent_type: "defender"`, `team_name: "review-{slug}"`, `model: "opus"`

  Prompt: "**Edit scope (sharply bounded).** Allowed: files named in an active Blocking finding's `Location: file:line`, and test files at paths corresponding to those files. Forbidden: the PRP at `docs/PRPs/{slug}.md` (it is the contract being compared against), `CLAUDE.md` (convention changes go through `/update-claude-md`), methodology files under `prompts/`, `.claude/commands/`, `.claude/hooks/`, `.claude/agents/`, and files outside the PR diff scope (not modified between `main` and `HEAD`). Every defender commit must touch only allowed-scope files — a post-commit checker (`scripts/check-defender-scope.sh`) enforces this before the review artifact is written.

  **Response protocol:** defend decisions with evidence (cite PRP tasks, plan review yield points, design docs). If a Blocking finding is valid, fix it immediately: edit the code, run tests, commit with conventional format (`fix(defender): <short description>`, ≤72 chars first line). If a non-Blocking finding is valid, acknowledge it — don't defend for the sake of defending. Never weaken tests to resolve a finding.

  **Your task:** Defend the implementation on this branch. Read the PRP at docs/PRPs/{slug}.md, plan review, and CLAUDE.md for context. Run `git diff main...HEAD` to understand all changes. Wait for findings from 'code-reviewer'. For each: defend with evidence, acknowledge, or fix (if Blocking). Send your response to 'code-reviewer' — they are waiting. Send final summary to team lead when done."

### 5. Assign tasks

Use `TaskUpdate` to assign Task A to "code-reviewer" and Task B to "defender".

### 5.5. Health check (after spawning)

After both teammates are spawned:
- Wait 60 seconds, then check `TaskList` — are tasks still in "not_started" status?
- If teammates haven't started within 90 seconds:
  1. Send `shutdown_request` to any active teammates
  2. Call `TeamDelete`
  3. **STOP** and report to user:
     "Team spawning failed. Teammates did not start within 90 seconds.
      This is likely Claude Code bug #32368 (model inheritance) or #24316 (agent definitions not loading).
      Options:
      - Retry: `/review-code-team`
      - Sequential fallback: `/review-code`
      - Check: `claude --version` (v2.1.46+ may fix this)"
  4. Do NOT fall back to solo review. Do NOT continue.

### 6. Actively manage the exchange

You are the **tech lead**, not a passive observer. You have the PRP, plan review, and CLAUDE.md. Participate:

- **Validate findings** — When the reviewer flags an issue, assess whether the severity is right. If the reviewer is inflating severity (e.g., calling a style issue Blocking), message both agents with your assessment.
- **Validate defenses** — When the defender pushes back on a finding, check if the defense holds. If the defender is dismissing a real issue, message them to take it seriously.
- **Verify fixes** — When the defender fixes a Blocking issue, spot-check the fix. Read the diff, run the tests yourself if needed.
- **Break deadlocks** — If the reviewer and defender disagree on severity and neither is moving, make the call as tech lead.
- **Check PRP compliance** — You have the PRP. If neither agent catches a deviation from spec, flag it yourself.

Track:
- Findings by severity (Blocking, Significant, Minor, Nit)
- Defender responses (defended, acknowledged, fixed)
- Whether all Blocking issues were resolved
- Any PRP deviations not caught by either agent

**Stall detection:** If the reviewer sends a finding and no defender response arrives within 3 minutes, message the defender asking for status. If still no response after 1 additional minute, message the reviewer to proceed and note the skipped response in the final artifact.

### 7. Synthesize code review artifact

After both teammates complete (check `TaskList`), prepare the artifact content but do NOT write it to disk yet.

### 8. Verify defender scope, then write and commit the artifact

Before writing the review artifact to `docs/reviews/code/{slug}-review.md`:

- Run `bash scripts/check-defender-scope.sh {slug}`.
- If it exits non-zero, ABORT: print the script's output, do NOT write the artifact, do NOT commit. Report the violation to the user and stop.

If the check passes, write the artifact:

```markdown
# Code Review: {slug}

## Verdict: APPROVED | REQUESTS CHANGES
## Reviewer Stance: Team — Code Reviewer + Implementation Defender
## Date: {today}
## Mode: Agent Team (message-gated code review)

## PRP Compliance
[Task-by-task comparison: implemented as specified, deviated, or missing]

## Issues Found

### 1. [Description]
**Category:** Security | Performance | Convention | Test Quality | Logic | ...
**Severity:** [initial] → [final after exchange]
**Location:** file:line
**Details:** [what was found]
**Defender Response:** [defended with evidence | acknowledged | fixed in commit {sha}]
**Resolution:** [resolved | open | downgraded]

### 2. ...

## What's Done Well
[Strengths noted during review — this is cooperative, not adversarial]

## Summary
[Overall assessment. Finding counts by severity. Whether all Blocking issues resolved.]
```

Commit message: `docs: dialectic code review for {slug} (team mode)`

### 9. Shutdown and cleanup

Send `shutdown_request` to both teammates. Wait for responses. Then call `TeamDelete`.

### 10. Report

- Verdict (APPROVED or REQUESTS CHANGES)
- Finding counts by final severity
- Key findings and how they were resolved
- If APPROVED: recommend `/security-audit-team` for full tier, otherwise create PR
- If REQUESTS CHANGES: list specific changes needed before re-review

## Validation

```bash
# Review artifact was written at the expected path
test -f "docs/reviews/code/{slug}-review.md"

# Required sections present
grep -qE "^## Verdict:" "docs/reviews/code/{slug}-review.md"
grep -qE "Defender Response" "docs/reviews/code/{slug}-review.md"

# Defender-scope enforcement succeeded
bash scripts/check-defender-scope.sh {slug}
```
