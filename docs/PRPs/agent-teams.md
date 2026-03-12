# PRP: Agent Teams Integration

## Source PRD: docs/PRDs/agent-teams.md
## Date: 2026-02-11
## Confidence Score: 7/10

## 1. Context Summary

Push Hands implements dialectical tension sequentially — the Proposer generates an artifact, then the Training Partner critiques it in a separate invocation. Claude Code's agent teams feature (February 2026) enables concurrent, multi-agent workflows where teammates communicate directly via `SendMessage`. This PRP adds an optional team mode that runs proposer and training partner stances as concurrent teammates, creating genuine real-time dialectical exchange.

Three new slash commands — `/review-plan-team`, `/execute-team`, `/security-audit-team` — augment (not replace) the existing sequential pipeline. Five custom agent definitions in `.claude/agents/` enforce stance constraints at the tool level.

The v1.3 roadmap item "Multi-Agent Orchestration" from `push-hands.md` is delivered by this feature.

## 2. Codebase Analysis

### Existing patterns

**Slash command structure.** All seven existing commands in `.claude/commands/` follow the same pattern:
- Line 1: stance declaration ("You are operating in the **X** stance...")
- Body: numbered instruction steps
- Arguments via `$ARGUMENTS`
- Artifact output to `docs/{PRDs,PRPs,reviews,audits}/`
- Commit with conventional message
- Recommend next step

Team commands must follow this same structural pattern, but with team orchestration steps added.

**AGENTS.md stance definitions** (`AGENTS.md:75-155`):
- Five stances with character, goal, constraints, and key behaviors
- Proposer: "Thorough, systematic, completion-oriented" — full capabilities
- Senior Training Partner: "Cannot write code — only test the plan's balance" — read-only
- Code Review Partner: "Reviews diff only — does not rewrite" — read-only
- Security Auditor: "Must provide proof of concept for HIGH+" — needs Bash for PoC
- Skeptical Client: "Demands proof" — pure analysis, read-only

**Artifact paths** (from `push-hands.md` Appendix B, line 724-731):
- PRPs: `docs/PRPs/<feature-slug>.md`
- Plan reviews: `docs/reviews/plans/<feature-slug>-review.md`
- Code reviews: `docs/reviews/code/<feature-slug>-review.md`
- Audits: `docs/audits/<feature-slug>-audit.md`

Team-mode commands produce artifacts at the **same paths** as their sequential equivalents, with the **same top-level sections** but additional team-specific fields that capture the dialectical exchange. Team-mode artifacts are strict supersets of sequential artifacts — any consumer that reads the standard fields (Verdict, Yield Points Found, etc.) will find them in the same position.

**Review artifact format** (from `review-plan.md:31-53` and `review-code.md:28-55`):
- Plan reviews: Verdict, Yield Points Found, What Holds Well, Summary
- Code reviews: Verdict, PRP Compliance, Issues Found, What's Done Well, Summary
- Audit reports follow `docs/audits/TEMPLATE.md`: Scope, Methodology, Findings Summary table, Detailed Findings, What Not Found, Remediation Priority

**Team-mode artifact additions** (fields added within the standard sections, not replacing them):
- Plan reviews add per-yield-point: Proposer Response, PRP Updated (yes/no), plus a `Mode: Agent Team` header
- Code reviews add per-issue: Resolution (how the issue was resolved during implementation, if applicable)
- Audit reports add per-finding: Initial Severity, Final Severity, Client Challenge, Auditor Defense, Resolution — capturing the full challenge-defense exchange trail

**CI workflow** (`.github/workflows/push-hands-review.yml`): Checks for artifact files by path. Team-mode artifacts land at the same paths, so CI needs no changes.

**setup.sh** (`scripts/setup.sh:28-37`): Creates `docs/` subdirectories and verifies key files. Needs to add `.claude/agents/` directory verification.

### Conventions to follow

- Commits: conventional format, max 72 chars first line
- Slugs: `^[a-z0-9]+(-[a-z0-9]+)*$`
- Known gotcha: GitHub Actions `${{ }}` — use `env:` for untrusted values
- Known gotcha: `grep -qE` checks all lines — use `head -1 | grep -qE` for single-line validation
- Custom agent files: YAML frontmatter + markdown body, stored in `.claude/agents/`

### Integration points

- `.claude/agents/` — new directory, agents loaded automatically by Claude Code at session start
- `.claude/commands/` — three new slash command files, no changes to existing seven
- `AGENTS.md` — add section mapping stances → agent definitions
- `CLAUDE.md` — add agent teams conventions
- `push-hands.md` — update roadmap, add team mode section and Appendix A entries
- `README.md` — add team mode mention
- `scripts/setup.sh` — verify `.claude/agents/` directory
- `.github/PULL_REQUEST_TEMPLATE.md` — add team review mode note

## 3. External Research

### Claude Code custom agent format

Custom agents are Markdown files in `.claude/agents/` with YAML frontmatter:

```yaml
---
name: agent-name
description: When Claude should delegate to this agent
tools: Read, Grep, Glob, Bash
model: inherit
---

System prompt body here.
```

Key fields:
- `name` (required): Identifier, lowercase + hyphens
- `description` (required): Natural language — Claude uses this to decide when to auto-delegate
- `tools`: Comma-separated allowlist (omit to inherit all tools)
- `model`: `sonnet`, `opus`, `haiku`, or `inherit` (default: `inherit`)

Available tool names: `Read`, `Write`, `Edit`, `Bash`, `Grep`, `Glob`, `WebFetch`, `WebSearch`, `Task`, `NotebookEdit`

### Claude Code agent teams API

**TeamCreate**: Creates team + shared task list at `~/.claude/teams/{name}/` and `~/.claude/tasks/{name}/`.

**Task tool with `team_name` + `name`**: Spawns a teammate. The `subagent_type` can reference custom agents by name. The `run_in_background: true` parameter runs teammates concurrently.

**TaskCreate/TaskList/TaskUpdate**: Shared task list. Tasks have `subject`, `description`, `activeForm`, `status`, `owner`, `blockedBy`. Status lifecycle: `pending` → `in_progress` → `completed`.

**SendMessage**: Teammates communicate directly. Types: `message` (DM), `broadcast`, `shutdown_request`, `shutdown_response`. Messages auto-deliver — no polling needed.

**TeamDelete**: Cleanup. Fails if active members remain — must shutdown first.

### Key constraints

- Teammates cannot spawn their own teams or subagents
- One team per session
- No session resumption for in-process teammates
- File conflicts if two agents edit the same file — partition write access
- Each teammate is a separate Claude instance — costs scale linearly with team size
- Experimental feature: requires `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in settings

### Design decision: artifact ownership

Review/audit artifacts are written by the **team lead** (the main session executing the slash command), not by individual teammates. This keeps critic agents (training-partner, code-reviewer, skeptical-client) truly read-only. Teammates communicate findings via `SendMessage`; the lead synthesizes into proper artifact format. The proposer agent in `/execute-team` DOES write code directly (it needs `Write`/`Edit` tools).

### Design decision: context window management

The team lead receives all teammate messages and must synthesize them into artifacts. For long exchanges (10+ messages), this creates context pressure. Mitigation: each teammate's **final summary message must be self-contained for artifact synthesis** — it must include all information the lead needs to write the artifact without referencing individual earlier messages. The lead synthesizes primarily from these summaries, not from reconstructing the full message history. This ensures artifact quality doesn't degrade as exchange length increases.

### Design decision: team naming

Teams use `{stage}-{slug}` naming:
- `plan-review-{slug}` for `/review-plan-team`
- `execute-{slug}` for `/execute-team`
- `audit-{slug}` for `/security-audit-team`

### Design decision: model selection

All agents default to `model: inherit` (use whatever the main session uses). The user guide documents that switching critic agents to `haiku` saves cost but may degrade critique quality. Users can edit agent definitions to change this.

## 4. Implementation Plan

### Task 1: Create custom agent definitions

**Files:** `.claude/agents/proposer.md`, `.claude/agents/training-partner.md`, `.claude/agents/code-reviewer.md`, `.claude/agents/security-auditor.md`, `.claude/agents/skeptical-client.md`

**Approach:**

Create `.claude/agents/` directory with five agent definition files. Each file has YAML frontmatter (name, description, tools, model) and a markdown body (complete system prompt).

**IMPORTANT — description field:** Agent descriptions must NOT describe the agent's general capability. Descriptions that match common tasks (e.g., "Reviews code diffs") will cause Claude Code to auto-delegate to the agent during sequential command execution, breaking existing commands. All descriptions must explicitly state the agent is for team-mode use only.

**`.claude/agents/proposer.md`:**
```markdown
---
name: proposer
description: Push Hands team-mode agent. Do not invoke directly — used by /review-plan-team and /execute-team.
tools: Read, Write, Edit, Bash, Grep, Glob
model: inherit
---

You are operating in the **Proposer** stance as a team member in a Push Hands agent team.

## Character
Thorough, systematic, completion-oriented.

## Constraints
- Read CLAUDE.md before writing any code — follow all project conventions
- Reference existing code patterns when implementing
- Log uncertainty — what you guessed, what needs human review
- Run validation gates after each task (lint, type check, tests)

## Team Communication
You are part of an agent team. You MUST use SendMessage to communicate with teammates and the team lead.
- Send progress updates to the team lead after completing each task
- When you receive feedback from a teammate, evaluate it honestly — accept valid concerns, push back on invalid ones with justification
- When responding to a teammate, always use SendMessage with type "message" and their name as recipient
- When all your work is complete, send a final summary to the team lead and mark your task as completed via TaskUpdate

## How You Work
1. Read your assigned task description carefully
2. Read the PRP and any referenced documents
3. Execute the work as specified
4. Commit each unit of work atomically with conventional commit format
5. Communicate results via SendMessage
```

**`.claude/agents/training-partner.md`:**
```markdown
---
name: training-partner
description: Push Hands team-mode agent. Do not invoke directly — used by /review-plan-team.
tools: Read, Grep, Glob, Bash
model: inherit
---

You are operating in the **Senior Training Partner** stance as a team member in a Push Hands agent team.

## Character
Patient, perceptive, structurally attuned. You sense where plans yield under pressure — before implementation exposes it.

## Constraints
- You CANNOT write code or modify files — you have no Write or Edit tools
- You can only test the plan's balance through calibrated pressure
- You must cite specific yield points with evidence (file paths, line numbers)
- You must check every file path and symbol reference against the actual codebase
- Assume the plan has at least 3 structural weaknesses

## What You Look For
- Missing edge cases: inputs, states, or conditions not handled
- Incorrect assumptions about existing code
- Underspecified tasks that lack detail for unambiguous execution
- Security implications
- Performance concerns: N+1 queries, unbounded loops, memory pressure
- Unnecessary complexity: "Does this need to exist?"
- Hidden coupling and unstated dependencies
- Test gaps

## Team Communication
You are part of an agent team. You MUST use SendMessage to communicate findings.
- Send each yield point as a separate message to the "proposer" teammate — include severity, evidence, and your specific concern
- After all yield points are sent, send a summary message to the team lead with your overall assessment
- If the proposer responds defending a point, evaluate their response and either accept it or press further
- When done, mark your task as completed via TaskUpdate
```

**`.claude/agents/code-reviewer.md`:**
```markdown
---
name: code-reviewer
description: Push Hands team-mode agent. Do not invoke directly — used by /execute-team.
tools: Read, Grep, Glob, Bash
model: inherit
---

You are operating in the **Code Review Partner** stance as a team member in a Push Hands agent team.

## Character
Detail-oriented, convention-aware, quality-sensing. You review code — you do not rewrite it.

## Constraints
- You CANNOT modify files — you have no Write or Edit tools
- You review diffs only — identify where balance breaks, do not fix it yourself
- You must compare implementation against the PRP specification task by task

## What You Check
- Spec mismatch: does the code match the PRP specification?
- Security vulnerabilities: injection, auth bypass, data exposure, SSRF, path traversal
- Performance issues: N+1 queries, unbounded loops, memory leaks, missing pagination
- Test quality: meaningful tests, not just coverage — do they cover edge cases?
- Convention violations per CLAUDE.md
- Dead code, commented-out code, unresolved TODOs
- Error handling appropriateness

## Team Communication
You are part of an agent team. You MUST use SendMessage to communicate findings.
- When you receive a commit SHA from the "proposer" teammate, review it using `git show {sha}` and `git diff {sha}~1..{sha}`
- NEVER use HEAD-relative commands (git diff HEAD~1) — another agent may be moving HEAD
- Send each finding as a message to the "proposer" with category, severity, location (file:line), and suggestion
- Categorize findings: Blocking (must fix before continuing), Significant, Minor, Nit
- After all commits are reviewed, send a final summary to the team lead
- When done, mark your task as completed via TaskUpdate
```

**`.claude/agents/security-auditor.md`:**
```markdown
---
name: security-auditor
description: Push Hands team-mode agent. Do not invoke directly — used by /security-audit-team.
tools: Read, Grep, Glob, Bash
model: inherit
---

You are operating in the **Security Auditor** stance (hard stance) as a team member in a Push Hands agent team.

## Character
Thorough, exploit-minded, severity-calibrated. You are trying to break things — this is a deliberate shift from cooperative push hands to adversarial security testing.

## Constraints
- You CANNOT modify files — you have no Write or Edit tools
- You must provide proof of concept or clear exploitation path for HIGH+ findings
- Use Bash to demonstrate exploits where possible
- Severity ratings: CRITICAL / HIGH / MEDIUM / LOW / INFO
- You must explicitly state what you did NOT examine

## Vulnerability Categories
Hunt across: authentication/authorization (bypass, escalation, IDOR), input validation (injection, XSS, path traversal), data exposure (PII leaks, verbose errors), configuration (secrets, insecure defaults), business logic (race conditions, abuse), dependencies (CVEs), cryptography (weak algorithms, key management)

## Finding Format
For each finding, send a message containing:
- **Severity:** CRITICAL / HIGH / MEDIUM / LOW / INFO
- **Category:** (from list above)
- **Affected code:** file:line
- **Description:** what the vulnerability is
- **Proof of Concept:** exploitation steps or code (REQUIRED for HIGH+)
- **Remediation:** specific fix
- **Effort:** XS / S / M / L / XL

## Team Communication
You are part of an agent team. You MUST use SendMessage to communicate findings.
- Send each finding as a separate, structured message to both the "client" teammate and the team lead as you discover them — do NOT batch all findings at the end
- When the "client" challenges a finding, evaluate their challenge and respond: defend your severity with evidence, or acknowledge the downgrade
- After all findings are sent and challenges resolved, send a final summary to the team lead
- When done, mark your task as completed via TaskUpdate
```

**`.claude/agents/skeptical-client.md`:**
```markdown
---
name: skeptical-client
description: Push Hands team-mode agent. Do not invoke directly — used by /security-audit-team.
tools: Read, Grep, Glob
model: inherit
---

You are operating in the **Skeptical Client** stance (hard stance) as a team member in a Push Hands agent team.

## Character
Budget-conscious, dubious, demands proof. You challenge inflated severity, catch false positives, and ensure the audit report is defensible.

## Constraints
- You CANNOT modify files or run commands — you have no Write, Edit, or Bash tools
- You cannot dismiss findings without technical justification
- You must challenge every HIGH+ finding for evidence quality
- You must question remediation effort estimates

## How You Challenge
For each finding you receive from the "auditor":
- **HIGH+ findings:** Is the PoC convincing? Could you actually exploit this? Is the severity justified given the deployment context?
- **All findings:** Is this a real risk or theoretical? Are there false positives? Is the effort estimate realistic?
- If a finding is valid but over-rated, recommend a specific downgrade with justification
- If a finding is solid, acknowledge it — don't challenge for the sake of challenging

## Team Communication
You are part of an agent team. You MUST use SendMessage to communicate.
- Read each finding message from the "auditor" as it arrives
- Send your challenge or acknowledgment back to the "auditor" for each finding
- After all findings are reviewed and the exchange has stabilized, send your final assessment to the team lead: which findings survived scrutiny, which were downgraded, which were dismissed, and your recommended priority order
- When done, mark your task as completed via TaskUpdate
```

**Tests:** Read each file and verify: (1) YAML frontmatter has valid syntax, (2) description says "Push Hands team-mode agent. Do not invoke directly", (3) tool restrictions match AGENTS.md constraints, (4) body includes Team Communication section with SendMessage instructions and TaskUpdate completion signal.
**Validation:** Read each file and confirm structure. Grep for "Do not invoke directly" across all five agent files to confirm auto-delegation prevention.

### Task 2: Create `/review-plan-team` slash command

**Files:** `.claude/commands/review-plan-team.md`

**Approach:**

This slash command orchestrates a concurrent plan review where a Training Partner critiques the PRP while a Proposer defends and revises it in real time.

**Scope clarification (deviation from PRD):** The PRD (Section 2, line 32) describes `/review-plan-team` as concurrent *generation* and review ("A Proposer teammate drafts the PRP while a Training Partner reads alongside"). This PRP narrows the scope to concurrent *review and revision* of an already-existing PRP. Rationale: (1) separating generation from review keeps `/generate-prp` as the single PRP creation path, avoiding two divergent generation workflows; (2) concurrent generation requires the Training Partner to review a document mid-creation, which is operationally fragile — partial documents trigger false yield points; (3) the dialectical value comes from challenging a complete plan, not an incomplete draft. The PRD's Section 2 and AC #2 ("producing a PRP") should be updated to reflect this narrower scope — this is flagged as a post-implementation PRD update.

The prompt structure:
1. **Stance declaration:** "You are orchestrating a concurrent Push Hands plan review using agent teams. You are the team lead."
2. **Parse arguments:** Extract PRP path from `$ARGUMENTS`. Derive slug from the PRP filename (strip `docs/PRPs/` prefix and `.md` suffix). Note: this command requires an existing PRP — generation still happens via `/generate-prp`. This command enhances the review phase with concurrent revision.
3. **Gather context:** Read the PRP. Read the source PRD (linked in PRP header). Read CLAUDE.md.
4. **Create team:** Call `TeamCreate` with `team_name: "plan-review-{slug}"`.
5. **Create tasks (NO dependencies — both start simultaneously):**
   - Task 1: "Analyze PRP for structural weaknesses" — description includes: read the PRP at `{path}`, read the source PRD, verify every file path and symbol reference against the actual codebase, identify at least 3 yield points, send each finding as a separate message to "proposer" with severity and evidence as you discover them (do not batch). **Final summary to team lead must be self-contained for artifact synthesis:** list every yield point with severity, evidence, the proposer's response, whether it was accepted/rejected, and any strengths identified. The team lead will use this summary directly — do not assume the lead has tracked individual messages.
   - Task 2: "Defend and revise PRP based on training partner feedback" — description includes: read the PRP at `{path}`, then wait for messages from "training-partner". As each concern arrives, evaluate it: revise the PRP where concerns are valid (using Edit tool), respond to the training partner explaining what was changed and why, reject invalid concerns with justification. **Final summary to team lead must be self-contained for artifact synthesis:** list every yield point received, your response (accepted/rejected with rationale), what was changed in the PRP and where, and any unresolved concerns. The team lead will use this summary directly.
   - **Convergence rule:** The exchange is complete when: (a) the training partner has sent all yield points AND a summary message to the team lead, AND (b) the proposer has responded to every yield point and sent a summary of revisions to the team lead. If a revision introduces a new issue that the training partner catches, the training partner may raise additional yield points, but the exchange MUST terminate after a maximum of 2 revision rounds (initial findings + one round of follow-up on revisions). After the second round, remaining concerns are documented in the review artifact as "open items for human review."
   - **No `blockedBy` between tasks.** Both agents start simultaneously. The Proposer reads the PRP for context, then waits for incoming messages. This creates genuine concurrent exchange rather than sequential phases.
6. **Spawn teammates (both in parallel):**
   - `Task(name: "training-partner", subagent_type: "training-partner", team_name: "plan-review-{slug}", run_in_background: true)` — prompt includes the PRP path, PRD path, and instructions to send findings via SendMessage to "proposer" as discovered
   - `Task(name: "proposer", subagent_type: "proposer", team_name: "plan-review-{slug}", run_in_background: true)` — prompt includes the PRP path and instructions: "Read the PRP for context. Then wait for messages from training-partner. Respond to each concern as it arrives."
7. **Assign tasks:** TaskUpdate to assign Task 1 to "training-partner" and Task 2 to "proposer".
8. **Monitor exchange:** Wait for teammate messages. As messages arrive, track the exchange — what yield points were found, which were addressed, which were disputed.
9. **Synthesize review artifact:** After both teammates complete (check TaskList), write `docs/reviews/plans/{slug}-review.md` using the standard review format:
   - Verdict: APPROVED if all structural yield points were addressed in the PRP revision; REQUESTS CHANGES if unresolved structural issues remain
   - Yield Points Found: each finding from the training partner, with the proposer's response and whether the PRP was revised
   - What Holds Well: strengths identified by both agents
   - Summary: overall assessment and whether the PRP is ready for execution
10. **Commit:** `docs: push hands plan review for {slug} (team mode)`
11. **Shutdown:** Send `shutdown_request` to both teammates, wait for responses, then `TeamDelete`.
12. **Report:** Verdict, key yield points, recommended next step (`/execute-prp` or `/execute-team` if APPROVED, revision guidance if REQUESTS CHANGES).

**Tests:** Invoke on the PRP at `docs/PRPs/agent-teams.md` (this feature's own PRP) to verify the orchestration flow produces a valid review artifact at the correct path.
**Validation:** Read the generated review artifact and verify it follows the standard format. Check that the PRP was modified by the proposer where training partner feedback was valid.

### Task 3: Create `/execute-team` slash command

**Files:** `.claude/commands/execute-team.md`

**Approach:**

This slash command orchestrates parallel implementation and review — the Proposer implements PRP tasks while the Code Reviewer reviews completed commits incrementally.

The prompt structure:
1. **Stance declaration:** "You are orchestrating message-gated implementation and review using agent teams. You are the team lead."
2. **Parse arguments:** Extract PRP path from `$ARGUMENTS`. Derive slug.
3. **Gather context:** Read the PRP and any plan review at `docs/reviews/plans/{slug}-review.md`. Read CLAUDE.md.
4. **Create team:** `TeamCreate` with `team_name: "execute-{slug}"`.
5. **Create tasks (NO dependencies — coordination via messages):**
   - Task 1: "Implement all PRP tasks with message-gated review" — description includes: read the PRP at `{path}`, incorporate plan review feedback if it exists, execute tasks in order per PRP specification. **Message-gated workflow for each task:** (a) implement the task, (b) run validation gates, (c) commit atomically with conventional format, (d) send a message to "code-reviewer" containing the **commit SHA** (from `git rev-parse HEAD`) and a summary of what was implemented, (e) **WAIT for a response from "code-reviewer" before starting the next task** — if the reviewer flags a Blocking issue, fix it and re-commit before proceeding. **Timeout recovery:** if no response from "code-reviewer" arrives within 2 minutes of sending a commit SHA, send a message to the team lead noting the gap and proceed to the next task. **Final summary to team lead must be self-contained for artifact synthesis:** for each PRP task, list the commit SHA, what was implemented, reviewer findings (with category and severity), how each finding was resolved, and any skipped reviews. If stuck (3+ failures), stop and message the team lead.
   - Task 2: "Review implementation commits incrementally" — description includes: read the PRP to understand the specification, read CLAUDE.md for conventions. Wait for messages from "proposer" containing commit SHAs. **For each commit SHA received:** review using `git show {sha}` and `git diff {sha}~1..{sha}` (NEVER use HEAD-relative commands — the proposer may be mid-commit). Check for spec mismatch, security vulnerabilities, performance issues, test quality, convention violations. Categorize each finding as Blocking/Significant/Minor/Nit. Send findings to "proposer" via SendMessage. **Always respond to the proposer** even if the commit looks clean (send an acknowledgment) — the proposer is waiting. **Final summary to team lead must be self-contained for artifact synthesis:** for each commit reviewed, list the SHA, PRP task number, findings (category, severity, location, suggestion), resolution, and overall verdict. Include any commits that were not reviewed due to timeouts.
   - **Git safety:** The message-gated pattern ensures only one agent operates on the working tree at a time. The Proposer implements and commits, then waits. The Reviewer reviews the committed state (immutable), then responds. No concurrent git operations.
6. **Spawn teammates (both in parallel):**
   - `Task(name: "proposer", subagent_type: "proposer", team_name: "execute-{slug}", run_in_background: true)` — prompt includes PRP path and message-gated workflow instructions
   - `Task(name: "code-reviewer", subagent_type: "code-reviewer", team_name: "execute-{slug}", run_in_background: true)` — prompt includes PRP path and SHA-based review instructions
7. **Assign tasks.**
8. **Monitor and detect stalls:** As messages arrive, track implementation progress and review findings. If the reviewer flags a Blocking issue, confirm the proposer addresses it before continuing. **Stall detection:** if the proposer reports sending a commit SHA but no reviewer response arrives within 3 minutes, send a message to the code-reviewer asking for status. If still no response after 1 additional minute, send a message to the proposer to proceed to the next task and note the skipped review in the final artifact.
9. **Synthesize code review artifact:** After both complete, write `docs/reviews/code/{slug}-review.md`:
   - Verdict: APPROVED or REQUESTS CHANGES
   - PRP Compliance: task-by-task comparison
   - Issues Found: from the code reviewer's messages, categorized and with resolutions (note which were caught during implementation vs. post-hoc)
   - What's Done Well: strengths noted during incremental review
   - Summary
10. **Post-implementation:** Update PRD status to IMPLEMENTED.
11. **Commit review artifact:** `docs: push hands code review for {slug} (team mode)`
12. **Shutdown and cleanup.**
13. **Report:** Implementation summary, review verdict, recommended next step.

**Tests:** This command is tested end-to-end when used on a real feature. Verify it produces both implementation commits and a code review artifact.
**Validation:** Check that the review artifact follows the standard code review format. Verify implementation commits follow conventional commit format.

### Task 4: Create `/security-audit-team` slash command

**Files:** `.claude/commands/security-audit-team.md`

**Approach:**

This slash command orchestrates a true dual-agent security audit. The Security Auditor and Skeptical Client are separate teammates that exchange messages directly — the most natural fit for agent teams in the push-hands methodology.

The prompt structure:
1. **Stance declaration:** "You are orchestrating a dual-agent Push Hands Security Audit using agent teams. You are the team lead."
2. **Derive context:** Extract slug from branch name (`feature/{slug}`). Run `git diff main...HEAD` and `git log main..HEAD --oneline` to scope the audit.
3. **Create team:** `TeamCreate` with `team_name: "audit-{slug}"`.
4. **Create tasks (NO dependencies — both start simultaneously, coordinate via messages):**
   - Task 1: "Conduct security audit of feature changes" — description includes: scope the audit (attack surface from the diff), hunt for vulnerabilities across all categories (auth, input validation, data exposure, config, business logic, dependencies, crypto), for each finding document severity/category/affected code/description/PoC/remediation/effort. **Send each finding as a structured message to both "client" and the team lead as you discover it — do NOT batch findings at the end.** Must provide PoC or clear exploitation path for HIGH+ findings. When the "client" challenges a finding, respond with evidence or acknowledge the downgrade. **Final summary to team lead must be self-contained for artifact synthesis:** list every finding with initial severity, category, affected code, description, PoC summary, the client's challenge, your defense/acknowledgment, final severity, remediation, and effort. Include what was NOT examined.
   - Task 2: "Challenge audit findings for defensibility" — description includes: begin by reading the diff and codebase for context. Then wait for findings from "auditor" to arrive via messages. **Challenge each finding as it arrives** — do not wait for all findings before starting challenges. For HIGH+ findings: is PoC convincing? Is severity justified? For all findings: real risk or theoretical? False positive? Effort estimate realistic? Send challenges to "auditor" directly (they will respond and defend). **Final assessment to team lead must be self-contained for artifact synthesis:** for each finding, list the initial severity, your challenge, the auditor's defense, and the final severity you recommend. Include which findings survived scrutiny, which were downgraded, and your recommended remediation priority order.
   - **Convergence rule:** Each finding follows a fixed protocol: auditor reports → client challenges → auditor defends or acknowledges → finding is CLOSED (final severity determined). No further challenges on a closed finding. The exchange is complete when: (a) the auditor has sent all findings AND a summary to the team lead, AND (b) every finding has been through one challenge-defense cycle and the client has sent their final assessment. This single-round-per-finding protocol prevents oscillation while still producing meaningful dialectical tension.
   - **No `blockedBy` between tasks.** Both agents start simultaneously. The client reads the codebase for context while the auditor begins hunting. As findings arrive via messages, the client challenges them in real time — genuine dialectical exchange, not sequential phases.
5. **Spawn teammates:**
   - `Task(name: "auditor", subagent_type: "security-auditor", team_name: "audit-{slug}", run_in_background: true)` — prompt includes diff context and scope
   - `Task(name: "client", subagent_type: "skeptical-client", team_name: "audit-{slug}", run_in_background: true)` — prompt includes instructions to wait for and challenge findings
6. **Assign tasks.**
7. **Monitor exchange:** Track findings as the auditor reports them. Track challenges as the client pushes back. Track resolutions from the auditor's defense.
8. **Synthesize audit report:** After both complete, write `docs/audits/{slug}-audit.md` using the template:
   - Scope: what was examined, what was NOT
   - Methodology: "Dual-agent team audit — Security Auditor and Skeptical Client operating as separate agents with direct peer messaging"
   - Findings Summary table: final severity (post-challenge), category, description, effort
   - Detailed Findings: each finding with the full auditor-client exchange trail — what was found, how it was challenged, whether severity was adjusted, final determination
   - What This Audit Did NOT Find: limitations
   - Remediation Priority: ordered list
9. **Commit:** `docs: security audit for {slug} (team mode)`
10. **Shutdown and cleanup.**
11. **Report:** Finding count by severity, CRITICAL/HIGH findings needing attention, remediation priority.

**Tests:** Invoke on the current feature branch to verify the orchestration produces a valid audit artifact.
**Validation:** Check that the audit report follows the template format. Verify that findings include the dialectical exchange between auditor and client.

### Task 5: Create user guide

**Files:** `docs/guides/agent-teams.md`

**Approach:**

Create `docs/guides/` directory if it doesn't exist, then write a comprehensive user guide covering:

1. **What is team mode?** Brief explanation — concurrent multi-agent execution using Claude Code agent teams. Augments the existing sequential pipeline. Same artifact types at the same paths, with additional fields capturing the dialectical exchange.

2. **Prerequisites:**
   - Claude Code with agent teams enabled
   - Add to Claude Code settings (`.claude/settings.json` or user settings):
     ```json
     {
       "env": {
         "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
       }
     }
     ```
   - Note: This feature is experimental and may change

3. **When to use team mode vs. sequential:**
   | Scenario | Recommended | Why |
   |---|---|---|
   | Quick review of a small PRP | `/review-plan` (sequential) | Lower cost, sufficient for simple plans |
   | Complex PRP with many integration points | `/review-plan-team` | Real-time feedback loop catches issues before PRP is finalized |
   | Simple implementation | `/execute-prp` then `/review-code` | Sequential is cheaper and sufficient |
   | Large implementation with many tasks | `/execute-team` | Incremental review catches issues per-commit, not post-hoc |
   | Security audit (any complexity) | `/security-audit-team` | Dual-agent dialectic produces more defensible reports |

4. **Cost implications:** Each teammate is a separate Claude instance. A team with 2 agents costs ~2-3x a single invocation. Budget accordingly. Team mode is opt-in precisely because of this cost. Tips for reducing cost: switch critic agents to `haiku` model in `.claude/agents/` definitions.

5. **Available commands:**
   - `/review-plan-team <prp-path>` — Concurrent plan review + revision. **Requires an existing PRP** (generated via `/generate-prp`). This command enhances the review phase — the Training Partner critiques while the Proposer revises in real time. It does NOT generate the PRP from scratch.
   - `/execute-team <prp-path>` — Message-gated implementation + incremental review. The Proposer implements each task, commits, sends the SHA to the Code Reviewer, and waits for feedback before continuing.
   - `/security-audit-team` — Dual-agent security audit with real-time peer challenge. The Auditor sends findings as discovered; the Client challenges each one immediately.

6. **Pipeline with team mode:**
   ```
   /generate-prp → /review-plan-team → /execute-team → /security-audit-team (full tier)
   ```
   Team commands slot into the same pipeline positions as their sequential equivalents. You can mix: use `/generate-prp` (sequential), then `/review-plan-team` (team), then `/execute-prp` (sequential).

7. **Customizing agent definitions:** Agent definitions live in `.claude/agents/`. Users can edit tools, model, and prompt body. Link to AGENTS.md for stance reference.

8. **Limitations:** Experimental feature, no session resumption, one team per session, file conflict risk (mitigated by read-only critic agents and message-gated execution in `/execute-team`).

**Tests:** Read the guide and verify it's accurate and complete.
**Validation:** Visual review of markdown rendering.

### Task 6: Update AGENTS.md with agent definition mapping

**Files:** `AGENTS.md`

**Approach:**

Add a new section after the existing stance definitions (after line 155) titled "## Agent Definitions for Team Mode":

```markdown
## Agent Definitions for Team Mode

When using team-mode commands (`/review-plan-team`, `/execute-team`, `/security-audit-team`), each stance is realized as a custom agent definition in `.claude/agents/`. These definitions enforce stance constraints at the tool level — the training partner literally cannot write files, not just prompt-instructed not to.

| Stance | Agent File | Tools | Key Constraint |
|--------|-----------|-------|----------------|
| Proposer | `.claude/agents/proposer.md` | Read, Write, Edit, Bash, Grep, Glob | Full access — implements code |
| Senior Training Partner | `.claude/agents/training-partner.md` | Read, Grep, Glob, Bash | No Write/Edit — cannot modify files |
| Code Review Partner | `.claude/agents/code-reviewer.md` | Read, Grep, Glob, Bash | No Write/Edit — reviews only |
| Security Auditor | `.claude/agents/security-auditor.md` | Read, Grep, Glob, Bash | No Write/Edit — Bash for PoC testing |
| Skeptical Client | `.claude/agents/skeptical-client.md` | Read, Grep, Glob | No Write/Edit/Bash — pure analysis |

Review and audit artifacts are written by the team lead (the main session), not by individual agents. Agents communicate findings via `SendMessage`.

See `docs/guides/agent-teams.md` for setup and usage.
```

**Tests:** Read updated AGENTS.md and verify the table matches the actual agent definitions from Task 1.
**Validation:** Verify markdown table renders correctly.

### Task 7: Update CLAUDE.md with agent teams conventions

**Files:** `CLAUDE.md`

**Approach:**

1. Add to the "Development Lifecycle Pipeline" section, after the existing tier diagram (after the `Full:` line):

```markdown
### Team Mode (optional, standard/full tiers only)

Agent teams run proposer and training partner stances concurrently using Claude Code agent teams. Same artifact types at the same paths, with additional team-mode fields capturing the dialectical exchange.

```
Team Review:  /review-plan-team → concurrent PRP review + revision
Team Execute: /execute-team → parallel implementation + incremental review
Team Audit:   /security-audit-team → dual-agent security audit with peer challenge
```

Requires: `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in Claude Code settings. See `docs/guides/agent-teams.md`.
```

2. Add to the "Conventions" section:
```markdown
- Agent definitions: `.claude/agents/<stance>.md` (proposer, training-partner, code-reviewer, security-auditor, skeptical-client)
- Team names: `{stage}-{slug}` (e.g., `audit-my-feature`)
```

3. Add to "Key Design Decisions":
```markdown
- **Team mode augments, not replaces.** Sequential commands remain unchanged. Team commands produce the same artifact types at the same paths, with additional fields capturing the dialectical exchange.
- **Critic agents are tool-restricted.** Training partner and code reviewer have no Write/Edit access. The "cannot write code" constraint is enforced at the tool level, not just the prompt level.
```

**Tests:** Read updated CLAUDE.md and verify new sections are consistent with the rest.
**Validation:** Visual review.

### Task 8: Update push-hands.md with team mode and roadmap

**Files:** `push-hands.md`

**Approach:**

1. **Update Section 12 Roadmap** (line 704-708): Change v1.3 from "Multi-Agent Orchestration" planned items to:
```markdown
### v1.3 — Multi-Agent Orchestration ✓
- Agent team mode for concurrent proposer/training partner workflows
- Custom agent definitions in `.claude/agents/` with tool-level stance enforcement
- Team-mode slash commands: `/review-plan-team`, `/execute-team`, `/security-audit-team`
- User guide at `docs/guides/agent-teams.md`
```

2. **Add new Section 4.3** (or subsection after the existing pipeline description) titled "Team Mode (Agent Teams)":
   - Brief explanation: team mode uses Claude Code agent teams to run stances concurrently
   - Three commands with brief descriptions
   - Note: same artifact paths as sequential mode
   - Note: opt-in, standard/full tiers only
   - Cost consideration
   - Reference to `docs/guides/agent-teams.md` for full details

3. **Update Appendix A quick reference table** (line 713-722): Add three new rows for team commands:
```markdown
| Concurrent Plan Review | `/review-plan-team <prp>` | Standard, Full | PRP path | Revised PRP + `docs/reviews/plans/<slug>.md` | Team: Proposer + Training Partner |
| Parallel Implementation | `/execute-team <prp>` | Standard, Full | PRP path | Source code + tests + `docs/reviews/code/<slug>.md` | Team: Proposer + Code Reviewer |
| Dual-Agent Security Audit | `/security-audit-team` | Full | Branch diff | `docs/audits/<slug>.md` | Team: Auditor + Client |
```

**Tests:** Read the updated sections and verify consistency with the rest of the document.
**Validation:** Visual review.

### Task 9: Update README.md with team mode

**Files:** `README.md`

**Approach:**

1. Add three rows to the "Slash Commands" table (after line 49):
```markdown
| `/review-plan-team <prp>` | Standard, Full | PRP path | Revised PRP + plan review | Team: Proposer + Training Partner |
| `/execute-team <prp>` | Standard, Full | PRP path | Code + incremental review | Team: Proposer + Code Reviewer |
| `/security-audit-team` | Full | Branch diff | Audit report | Team: Auditor + Client |
```

2. Add a brief "Team Mode" subsection under "Customization" (after line 57):
```markdown
- **Team mode:** Run proposer and training partner stances concurrently using agent teams. Requires `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`. See `docs/guides/agent-teams.md`.
```

**Tests:** Read updated README and verify the table renders correctly.
**Validation:** Visual review of markdown.

### Task 10: Update scripts/setup.sh to verify agent definitions

**Files:** `scripts/setup.sh`

**Approach:**

1. After the existing "Create directory structure if missing" block (line 36-37), add:
```bash
mkdir -p .claude/agents
```

2. After the existing template file verification loop (line 53-59), add a new verification block:
```bash
echo "Checking agent definitions..."
AGENTS_MISSING=0
for agent in proposer training-partner code-reviewer security-auditor skeptical-client; do
    if [ -f ".claude/agents/${agent}.md" ]; then
        echo "  ✓ .claude/agents/${agent}.md"
    else
        echo "  ✗ .claude/agents/${agent}.md — MISSING"
        AGENTS_MISSING=1
    fi
done

if [ "$AGENTS_MISSING" -eq 1 ]; then
    echo "  Some agent definitions are missing. Team mode commands may not work."
    echo "  Agent definitions are optional — sequential commands work without them."
fi
echo ""
```

**Tests:** Run `bash -n scripts/setup.sh` to verify syntax. Run `./scripts/setup.sh` to verify it completes without error and reports agent definition status.
**Validation:** `bash -n scripts/setup.sh`

### Task 11: Update PR template with team mode note

**Files:** `.github/PULL_REQUEST_TEMPLATE.md`

**Approach:**

Add a note to the "Linked Artifacts" section (after line 15) indicating that team-mode artifacts use the same paths:
```markdown
<!-- Team mode (/review-plan-team, /execute-team, /security-audit-team) produces artifacts at the same paths -->
```

Add to the "Standard and Full tiers" checklist (after line 42):
```markdown
- [ ] If team mode used: agent team shutdown and cleanup completed
```

**Tests:** Visual review of the template.
**Validation:** Ensure valid markdown.

## 5. Validation Gates

```bash
# Verify agent definition files exist and have valid frontmatter
for agent in proposer training-partner code-reviewer security-auditor skeptical-client; do
    test -f ".claude/agents/${agent}.md" && echo "PASS: ${agent}.md exists" || echo "FAIL: ${agent}.md missing"
done

# Verify slash command files exist
for cmd in review-plan-team execute-team security-audit-team; do
    test -f ".claude/commands/${cmd}.md" && echo "PASS: ${cmd}.md exists" || echo "FAIL: ${cmd}.md missing"
done

# Verify user guide exists
test -f "docs/guides/agent-teams.md" && echo "PASS: guide exists" || echo "FAIL: guide missing"

# Shell script syntax check
bash -n scripts/setup.sh

# Verify existing commands are unchanged
git diff main -- .claude/commands/review-plan.md .claude/commands/execute-prp.md .claude/commands/security-audit.md .claude/commands/review-code.md .claude/commands/generate-prp.md .claude/commands/new-feature.md .claude/commands/update-claude-md.md | wc -l
# Should output 0 — no changes to existing commands

# Verify team-mode artifact formats are strict supersets of sequential formats
# Cross-check: every field in sequential review-plan.md artifact template must appear in review-plan-team.md
# Cross-check: every field in sequential review-code.md artifact template must appear in execute-team.md
# Cross-check: every field in sequential security-audit.md artifact template must appear in security-audit-team.md
# Manual verification — read each pair and confirm team format includes all sequential fields plus team-specific additions
```

## 6. Rollback Plan

All changes are on the `feature/agent-teams` branch. Main is untouched.

1. `git checkout main` — return to main.
2. `git branch -D feature/agent-teams` — delete the branch entirely.

Since this adds new files and modifies only documentation (no application logic), the risk is minimal. The rollback is "don't merge."

If merged and problems arise:
- Delete `.claude/agents/` directory — removes agent definitions, disables auto-delegation
- Delete team-mode slash commands — removes team commands, sequential commands remain functional
- Revert documentation changes via `git revert`

## 7. Uncertainty Log

1. **Slash command orchestration complexity.** The team-mode slash commands are significantly more complex than sequential commands. They instruct the main session to create teams, spawn teammates, manage tasks, monitor messages, and synthesize artifacts — all within a single command prompt. I have not tested whether Claude Code reliably follows multi-step orchestration prompts of this complexity. If the orchestration is unreliable, a fallback would be a simpler pattern where the lead does more work and teammates are more narrowly scoped. **Mitigation added by plan review:** teammate final summaries are now required to be self-contained for artifact synthesis, reducing the lead's dependency on tracking individual messages and limiting context pressure. **Confidence: 5/10 → 6/10.**

2. **Custom agent `subagent_type` reference.** The PRP assumes that custom agents defined in `.claude/agents/` can be referenced by name as `subagent_type` values in the Task tool (e.g., `subagent_type: "training-partner"`). The research confirms this is supported, but I haven't verified it against the actual Task tool parameter validation. If custom names aren't accepted, the fallback is to use `subagent_type: "general-purpose"` and embed the stance prompt in the Task tool's `prompt` parameter. **Confidence: 7/10.**

3. **Peer-to-peer message exchange quality.** The dialectical value of team mode depends on teammates engaging in productive multi-round exchange (auditor reports finding → client challenges → auditor defends → resolution). The research confirms `SendMessage` supports this, but the quality of the exchange depends on prompt engineering. If exchanges are shallow or one-sided, the team mode offers less value over sequential mode. **Confidence: 6/10.**

4. ~~**Task dependency timing in `/review-plan-team`.**~~ **Resolved by plan review.** All three team commands now start both agents simultaneously with no `blockedBy` dependencies. Coordination happens via messages, not task completion ordering. The Proposer/Client reads the PRP/diff for context then waits for incoming messages. **Confidence: 8/10.**

5. **Experimental feature stability.** Claude Code agent teams requires `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`. The feature is experimental and the API surface may change. If tool names, parameters, or behaviors change, the slash commands will need updating. This is documented but represents ongoing maintenance risk. **Confidence: 8/10 that it works today, lower for long-term stability.**

6. ~~**Agent definition `description` field and auto-delegation.**~~ **Resolved by plan review.** All agent descriptions now say "Push Hands team-mode agent. Do not invoke directly — used by /command-name." This prevents Claude Code from auto-delegating to these agents during sequential command execution. **Confidence: 9/10.**

7. ~~**Message-gated execution in `/execute-team`.**~~ **Partially resolved by plan review.** Three layers of mitigation now exist: (a) the Reviewer's prompt instructs "always respond even if clean," (b) the Proposer has a 2-minute timeout to proceed if no response arrives, (c) the team lead monitors for stalls and can intervene. Prompt compliance still isn't guaranteed, but the timeout and lead monitoring prevent indefinite blocking. **Confidence: 7/10.**
