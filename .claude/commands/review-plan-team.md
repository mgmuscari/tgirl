You are orchestrating a **concurrent Push Hands plan review** using agent teams. You are the team lead.

This command requires an existing PRP — generation still happens via `/generate-prp`. This command enhances the review phase: a Training Partner critiques the PRP while a Proposer defends and revises it in real time.

## Instructions

Review the PRP at: $ARGUMENTS

### 1. Parse and gather context

- Extract the PRP path from `$ARGUMENTS`. Derive the slug from the filename (strip `docs/PRPs/` prefix and `.md` suffix).
- Read the PRP thoroughly.
- Read the source PRD linked in the PRP header.
- Read CLAUDE.md for project conventions.

### 2. Create the team

- Call `TeamCreate` with `team_name: "plan-review-{slug}"`.

### 3. Create tasks (no dependencies — both start simultaneously)

Create two tasks via `TaskCreate`:

**Task A: "Analyze PRP for structural weaknesses"**
Description: You are the Senior Training Partner. Read the PRP at `{path}` and the source PRD. Verify every file path and symbol reference against the actual codebase. Identify at least 3 yield points — missing edge cases, incorrect assumptions, underspecified tasks, security implications, performance concerns, unnecessary complexity, hidden coupling, test gaps. Send each yield point as a separate message to the "proposer" teammate with severity and evidence as you discover them. After all yield points are sent, send a summary to the team lead. **Final summary to team lead must be self-contained for artifact synthesis:** list every yield point with severity, evidence, the proposer's response, whether it was accepted/rejected, and any strengths identified. The team lead will use this summary directly — do not assume the lead has tracked individual messages.

**Task B: "Defend and revise PRP based on training partner feedback"**
Description: You are the Proposer. Read the PRP at `{path}` for context. Then wait for messages from "training-partner". As each concern arrives: evaluate it honestly, revise the PRP where the concern is valid (use Edit tool), respond to "training-partner" explaining what you changed and why. Reject invalid concerns with justification. **Final summary to team lead must be self-contained for artifact synthesis:** list every yield point received, your response (accepted/rejected with rationale), what was changed in the PRP and where, and any unresolved concerns. The team lead will use this summary directly.

**Convergence rule:** The exchange is complete when: (a) the training partner has sent all yield points AND a summary message to the team lead, AND (b) the proposer has responded to every yield point and sent a summary of revisions to the team lead. If a revision introduces a new issue that the training partner catches, the training partner may raise additional yield points, but the exchange MUST terminate after a maximum of 2 revision rounds (initial findings + one round of follow-up on revisions). After the second round, remaining concerns are documented in the review artifact as "open items for human review."

**Do NOT use `addBlockedBy`.** Both tasks start simultaneously. The Proposer reads the PRP then waits for incoming messages — this creates genuine concurrent exchange.

### 4. Spawn teammates (both in parallel)

Spawn both teammates using the Agent tool with `run_in_background: true` and `model: "opus"`:

- `name: "training-partner"`, `subagent_type: "training-partner"`, `team_name: "plan-review-{slug}"`, `model: "opus"`
  Prompt: _(inline stance — required because .claude/agents/*.md definitions don't load for team members, see Claude Code bug #24316)_

  "You are the **Senior Training Partner** — patient, perceptive, structurally attuned. You sense where plans yield under pressure — before implementation exposes it. You CANNOT write code or modify files — you have no Write or Edit tools. Your tools are: Read, Grep, Glob, Bash.

  **Your constraints:**
  - You can only test the plan's balance through calibrated pressure
  - You must cite specific yield points with evidence (file paths, line numbers)
  - You must check every file path and symbol reference against the actual codebase
  - Assume the plan has at least 3 structural weaknesses

  **What you look for:** Missing edge cases, incorrect assumptions about existing code, underspecified tasks, security implications, performance concerns, unnecessary complexity, hidden coupling, test gaps.

  **Your task:** Review the PRP at {path}. The source PRD is at {prd_path}. Read CLAUDE.md for project conventions. Find at least 3 structural weaknesses. Send each yield point as a separate message to 'proposer' with severity and evidence. After all yield points, send a summary to the team lead."

- `name: "proposer"`, `subagent_type: "proposer"`, `team_name: "plan-review-{slug}"`, `model: "opus"`
  Prompt: _(inline stance — required because .claude/agents/*.md definitions don't load for team members, see Claude Code bug #24316)_

  "You are the **Proposer** — thorough, systematic, completion-oriented. You have full tool access (Read, Write, Edit, Bash, Grep, Glob).

  **Your constraints:**
  - Read CLAUDE.md before any work — follow all project conventions
  - Evaluate feedback honestly — accept valid concerns, push back on invalid ones with justification
  - When revising the PRP, use the Edit tool to make specific changes

  **Your task:** Defend and revise the PRP at {path}. Read the PRP for context. Then wait for messages from 'training-partner'. For each concern: if valid, edit the PRP and tell training-partner what you changed; if invalid, explain why. When done, send a summary of all revisions to the team lead."

### 5. Assign tasks

Use `TaskUpdate` to assign Task A to "training-partner" and Task B to "proposer".

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
      - Retry: `/review-plan-team {path}`
      - Sequential fallback: `/review-plan {path}`
      - Check: `claude --version` (v2.1.46+ may fix this)"
  4. Do NOT fall back to solo review. Do NOT continue.

### 6. Monitor the exchange

Wait for messages from both teammates. Track:
- What yield points were found (from training-partner messages)
- How the proposer responded (accepted/revised or rejected with justification)
- Whether the exchange converged (both agents complete their tasks)

### 7. Synthesize review artifact

After both teammates complete (check `TaskList`), write `docs/reviews/plans/{slug}-review.md`:

```markdown
# Plan Review: {slug}

## Verdict: APPROVED | REQUESTS CHANGES
## Reviewer Stance: Team — Senior Training Partner + Proposer
## Date: {today}
## Mode: Agent Team (concurrent review + revision)

## Yield Points Found

### 1. [Description]
**Severity:** Structural | Moderate | Minor
**Evidence:** [from training partner's message]
**Proposer Response:** [accepted and revised | rejected with justification]
**PRP Updated:** Yes/No

### 2. ...

## What Holds Well
[Strengths identified by both agents]

## Summary
[Overall assessment. APPROVED if all structural yield points were addressed in the PRP revision. REQUESTS CHANGES if unresolved structural issues remain.]
```

### 8. Commit

Message: `docs: push hands plan review for {slug} (team mode)`

### 9. Shutdown and cleanup

Send `shutdown_request` to both teammates. Wait for responses. Then call `TeamDelete`.

### 10. Report

- Verdict (APPROVED or REQUESTS CHANGES)
- Key yield points and how they were resolved
- If APPROVED: recommend `/execute-prp` or `/execute-team`
- If REQUESTS CHANGES: list what needs fixing
