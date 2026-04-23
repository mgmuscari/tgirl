You are orchestrating **message-gated implementation and review** using agent teams. You are the team lead.

The Proposer implements PRP tasks while the Code Reviewer reviews each commit incrementally. The Proposer waits for reviewer feedback after each commit before starting the next task — this prevents git race conditions and catches issues early.

## Instructions

Execute the PRP at: $ARGUMENTS

### 1. Parse and gather context

- Extract the PRP path from `$ARGUMENTS`. Derive the slug from the filename (strip `docs/PRPs/` prefix and `.md` suffix).
- Read the PRP and any plan review at `docs/reviews/plans/{slug}-review.md`. Incorporate review feedback.
- Read CLAUDE.md for project conventions.

### 2. Create the team

- Call `TeamCreate` with `team_name: "execute-{slug}"`.

### 3. Create tasks (no dependencies — coordination via messages)

Create two tasks via `TaskCreate`:

**Task A: "Implement all PRP tasks with message-gated review"**
Description: You are the Proposer. Read the PRP at `{path}`. Incorporate plan review feedback if it exists at `docs/reviews/plans/{slug}-review.md`. Read CLAUDE.md for conventions. Execute tasks in order per PRP specification. **Message-gated workflow for each task:** (a) implement the task, (b) run validation gates, (c) commit atomically with conventional format, (d) send a message to "code-reviewer" containing the commit SHA (from `git rev-parse HEAD`) and a summary of what was implemented, (e) WAIT for a response from "code-reviewer" before starting the next task — if the reviewer flags a Blocking issue, fix it and re-commit before proceeding. **Timeout recovery:** if no response from "code-reviewer" arrives within 2 minutes of sending a commit SHA, send a message to the team lead noting the gap and proceed to the next task. **Final summary to team lead must be self-contained for artifact synthesis:** for each PRP task, list the commit SHA, what was implemented, reviewer findings (with category and severity), how each finding was resolved, and any skipped reviews. The team lead will use this summary directly — do not assume the lead has tracked individual messages. If stuck (3+ failures on the same task), stop and message the team lead.

**Task B: "Review implementation commits incrementally"**
Description: You are the Code Review Partner. Read the PRP at `{path}` to understand the specification. Read CLAUDE.md for conventions. Wait for messages from "proposer" containing commit SHAs. For each SHA: review using `git show {sha}` and `git diff {sha}~1..{sha}` — NEVER use HEAD-relative commands. Check for spec mismatch, security vulnerabilities, performance issues, test quality, convention violations. Categorize findings: Blocking (must fix), Significant, Minor, Nit. Send findings to "proposer". ALWAYS respond even if the commit looks clean — the proposer is waiting. **Final summary to team lead must be self-contained for artifact synthesis:** for each commit reviewed, list the SHA, PRP task number, findings (category, severity, location, suggestion), resolution, and overall verdict. Include any commits that were not reviewed due to timeouts. The team lead will use this summary directly.

**Do NOT use `addBlockedBy`.** Both agents start simultaneously. The Code Reviewer reads the PRP for context, then waits for commit SHAs via messages.

### 4. Spawn teammates (both in parallel)

Spawn both teammates using the Agent tool with `run_in_background: true` and `model: "opus"`:

- `name: "proposer"`, `subagent_type: "proposer"`, `team_name: "execute-{slug}"`, `model: "opus"`

  Prompt: "**TDD is mandatory.** For each PRP task, follow this loop:
  1. RED: Write test(s) that specify expected behavior. Run the test command — verify tests FAIL. If tests pass before implementation, they are too weak — rewrite them.
  2. GREEN: Implement minimum code to make tests pass. Run the test command — verify tests PASS. If tests still fail, fix implementation (never weaken tests). Never mock to make tests pass.
  3. REFACTOR: Clean up if needed (tests must still pass).
  4. COMMIT: Test + implementation together, one atomic commit with conventional format.

  **Your task:** Implement the PRP at {path}. Read the PRP, plan review (if exists at docs/reviews/plans/{slug}-review.md), and CLAUDE.md. For each PRP task: follow the TDD loop above, then send the commit SHA (from `git rev-parse HEAD`) and a summary to 'code-reviewer' via SendMessage, and WAIT for their response before starting the next task. If they flag a Blocking issue, fix it first. Timeout: if no response within 2 minutes, message the team lead and proceed. Send final summary to team lead when done — include for each task: commit SHA, what was implemented, reviewer findings, resolutions, and skipped reviews."

- `name: "code-reviewer"`, `subagent_type: "code-reviewer"`, `team_name: "execute-{slug}"`, `model: "opus"`

  Prompt: "**Your task:** Review implementation of the PRP at {path}. Read the PRP and CLAUDE.md for context. Wait for commit SHAs from 'proposer' via messages. For each SHA: review the commit using `git show {sha}` and `git diff {sha}~1..{sha}` (NEVER HEAD-relative commands). Check for spec mismatch, security vulnerabilities, performance issues, test quality, convention violations. Categorize findings: Blocking (must fix), Significant, Minor, Nit. Send findings to 'proposer'. ALWAYS respond even for clean commits — the proposer is waiting. Send final summary to team lead when done — include for each commit: SHA, PRP task number, findings, resolution, and verdict."

### 5. Assign tasks

Use `TaskUpdate` to assign Task A to "proposer" and Task B to "code-reviewer".

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
      - Retry: `/execute-team {path}`
      - Sequential fallback: `/execute-prp {path}` then `/review-code`
      - Check: `claude --version` (v2.1.46+ may fix this)"
  4. Do NOT fall back to solo implementation. Do NOT continue.

### 6. Actively manage progress

You are the **tech lead**, not a passive observer. You have full architectural context, the PRP, the plan review, and CLAUDE.md. Participate:

- **Validate review findings** — When the code-reviewer flags an issue, assess whether it's blocking, significant, or noise. Message the proposer with your assessment if the reviewer's severity seems wrong.
- **Intervene on workarounds** — If the proposer stubs something out, defers a decision, or implements a workaround instead of the real fix, call it out. No "fix later" shims land silently.
- **Unblock the team** — If the proposer hits a technical obstacle (hook not firing, type error, missing API, ambiguous spec), investigate yourself and provide the answer rather than letting them spin.
- **Verify quality** — Spot-check commits. Read diffs for critical tasks. Don't rubber-stamp summaries.
- **Enforce TDD** — If a commit has implementation without tests, or tests that don't fail before implementation, flag it.

Track:
- Implementation progress (which PRP tasks are complete)
- Review findings (categorized by severity)
- Whether Blocking issues were resolved
- Deviations from the PRP specification

**Stall detection:** if the proposer reports sending a commit SHA but no reviewer response arrives within 3 minutes, send a message to the code-reviewer asking for status. If still no response after 1 additional minute, send a message to the proposer to proceed to the next task and note the skipped review in the final artifact.

### 7. Synthesize code review artifact

After both teammates complete (check `TaskList`), write `docs/reviews/code/{slug}-review.md`:

```markdown
# Code Review: {slug}

## Verdict: APPROVED | REQUESTS CHANGES
## Reviewer Stance: Team — Proposer + Code Review Partner
## Date: {today}
## Mode: Agent Team (message-gated incremental review)

## PRP Compliance
[Task-by-task comparison: implemented as specified, deviated, or missing]

## Issues Found

### 1. [Description]
**Category:** Security | Performance | Convention | Test Quality | Logic | ...
**Severity:** Blocking | Significant | Minor | Nit
**Location:** file:line
**Details:** [what was found]
**Resolution:** [how it was resolved during implementation, or still open]

### 2. ...

## What's Done Well
[Strengths noted during incremental review]

## Summary
[Overall assessment]
```

### 8. Post-implementation

Update the PRD status to IMPLEMENTED.

### 9. Commit review artifact

Message: `docs: dialectic code review for {slug} (team mode)`

### 10. Shutdown and cleanup

Send `shutdown_request` to both teammates. Wait for responses. Then call `TeamDelete`.

### 11. Report

- Implementation summary (tasks completed, commits made)
- Review verdict
- Key findings and resolutions
- Recommended next step: `/security-audit` or `/security-audit-team` for full tier, otherwise create PR
