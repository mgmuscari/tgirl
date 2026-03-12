# PRD: Agent Teams Integration

## Status: IMPLEMENTED
## Author: agent (Proposer)
## Date: 2026-02-11
## Branch: feature/agent-teams

## 1. Problem Statement

Push Hands currently implements dialectical tension *sequentially* — the Proposer generates an artifact, then the Training Partner critiques it in a separate invocation. This works, but has structural limitations:

**Anchoring bias.** By the time the Training Partner reviews a PRP or codebase, the Proposer's framing is already locked in. The reviewer is reacting to a finished artifact rather than sensing weakness as it forms. Sequential review catches *expressed* flaws but misses *structural* ones that a concurrent partner would surface earlier.

**No real-time dialogue.** The push hands metaphor implies continuous contact and feedback — sensing where structure yields *during* movement, not after. Current workflow is more like sparring with rounds: propose, step back, review, step back, revise. True tui shou has no rounds.

**Scaling wall.** For complex features, the standard pipeline is strictly serial. A security auditor cannot begin threat modeling while the plan reviewer is still examining task ordering. The full-tier pipeline especially suffers — security audit happens last, when the cost of change is highest.

**v1.3 roadmap gap.** The canonical specification (`push-hands.md` Section 12) lists "Multi-Agent Orchestration" as a v1.3 goal: parallel worktree management, subagent delegation, context window management. Claude Code's agent teams feature — launched February 2026 — provides the infrastructure to deliver this roadmap item now.

**Who needs this:** Solo developers and small teams using the push-hands template who want higher-fidelity adversarial dynamics without adding human reviewers. Particularly valuable for full-tier workflows where security-sensitive changes benefit from concurrent, multi-perspective analysis.

## 2. Proposed Solution

Add an **optional team mode** to the push-hands template that uses Claude Code agent teams to run proposer and training partner stances as concurrent teammates rather than sequential invocations.

### Core concept

Team mode does not replace the existing sequential workflow — it augments it. When enabled, specific pipeline stages spawn a team where agents in different stances work concurrently and communicate directly via `SendMessage`. The team lead orchestrates, but proposer and critic teammates can message each other, creating genuine dialectical exchange.

### Three integration points

1. **`/review-plan-team`** — Concurrent plan generation and review. A Proposer teammate drafts the PRP while a Training Partner teammate reads alongside and sends structural concerns in real time. The Proposer revises before the artifact is "finished."

2. **`/execute-team`** — Parallel implementation with live review. A Proposer teammate implements tasks while a Code Review Partner teammate reviews completed commits as they land, rather than reviewing the entire diff post-hoc.

3. **`/security-audit-team`** — True dual-agent security audit. The Security Auditor and Skeptical Client run as separate teammates that message each other directly — the Auditor reports findings, the Client challenges them, and the exchange produces a more defensible report than the current single-invocation two-phase approach.

### Custom agent definitions

Each push-hands stance becomes a custom agent definition in `.claude/agents/`, giving teammates the correct tool restrictions, model selection, and behavioral prompts. The Senior Training Partner agent, for example, would be configured as read-only (no Edit/Write tools), enforcing the "cannot write code" constraint at the tool level rather than relying solely on prompt compliance.

### Opt-in activation

Team mode requires explicit opt-in:
- Per-invocation: `--team` flag on slash commands (e.g., `/review-plan-team`)
- The existing sequential commands remain unchanged and are always available
- No changes to light tier (team mode is standard/full only)

## 3. Architecture Impact

### New files

| Path | Purpose |
|------|---------|
| `.claude/agents/proposer.md` | Custom agent: Proposer stance with full tool access |
| `.claude/agents/training-partner.md` | Custom agent: Senior Training Partner, read-only tools |
| `.claude/agents/code-reviewer.md` | Custom agent: Code Review Partner, read-only tools |
| `.claude/agents/security-auditor.md` | Custom agent: Security Auditor, full tools (needs PoC) |
| `.claude/agents/skeptical-client.md` | Custom agent: Skeptical Client, read-only tools |
| `.claude/commands/review-plan-team.md` | Slash command: concurrent plan review |
| `.claude/commands/execute-team.md` | Slash command: parallel execute + review |
| `.claude/commands/security-audit-team.md` | Slash command: dual-agent security audit |
| `docs/guides/agent-teams.md` | User guide: when and how to use team mode |

### Modified files

| Path | Change |
|------|--------|
| `AGENTS.md` | Add section mapping stances to `.claude/agents/` definitions |
| `CLAUDE.md` | Add agent teams conventions and known gotchas |
| `push-hands.md` | Update roadmap (v1.3 → delivered), add team mode section |
| `README.md` | Add team mode to quick-start |
| `scripts/setup.sh` | Verify `.claude/agents/` directory and agent definitions exist |
| `.github/PULL_REQUEST_TEMPLATE.md` | Add team mode checkbox for reviews |

### Dependencies

- Claude Code with agent teams enabled (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in settings)
- No new package dependencies — agent teams are a Claude Code built-in feature
- Custom agents use `.claude/agents/*.md` with YAML frontmatter (native Claude Code convention)

### Data model

No persistent data model changes. Teams are ephemeral — created at pipeline stage start, destroyed at stage end. Task lists live in `~/.claude/tasks/{team-name}/` and are cleaned up by `TeamDelete`. Review artifacts are written to the existing `docs/reviews/` and `docs/audits/` paths.

## 4. Acceptance Criteria

1. Five custom agent definitions exist in `.claude/agents/` corresponding to the five stances in AGENTS.md, each with correct tool restrictions (training partner and skeptical client are read-only; auditor has full tools)
2. `/review-plan-team` spawns a team where a Proposer teammate generates/refines a PRP and a Training Partner teammate concurrently reviews, with direct message exchange between them, producing a PRP and plan review artifact
3. `/execute-team` spawns a team where a Proposer teammate implements PRP tasks and a Code Review Partner teammate reviews completed commits incrementally, producing implementation commits and a code review artifact
4. `/security-audit-team` spawns a team where a Security Auditor teammate and Skeptical Client teammate conduct a dialectical audit, exchanging direct messages to challenge/defend findings, producing an audit artifact
5. All team-mode commands produce the same artifact types in the same paths as their sequential equivalents (PRPs in `docs/PRPs/`, reviews in `docs/reviews/`, audits in `docs/audits/`)
6. Existing sequential commands (`/review-plan`, `/execute-prp`, `/review-code`, `/security-audit`) remain unchanged and fully functional
7. Team mode is opt-in — no behavioral change for users who do not invoke team commands
8. `scripts/setup.sh` verifies `.claude/agents/` directory and agent definitions
9. `push-hands.md` roadmap updated to reflect v1.3 delivery
10. User guide at `docs/guides/agent-teams.md` documents when to use team mode vs. sequential mode, setup requirements, and cost implications

## 5. Risk Assessment

### High risk

- **Token cost explosion.** Each teammate is a separate Claude instance. A 3-agent team costs ~3x a solo invocation. Users must understand the cost tradeoff before opting in. Mitigation: clear documentation of cost implications, team mode is opt-in only, use haiku model for read-only agents where possible.

- **File conflict between teammates.** Two agents editing the same file produces overwrites. Mitigation: Only the Proposer agent has write access in review/audit teams. In `/execute-team`, the reviewer is read-only. Architectural constraint enforced at the agent definition level.

### Medium risk

- **Feature instability.** Agent teams is experimental and feature-gated. The API surface may change. Mitigation: team-mode commands are separate from sequential commands, so breakage is isolated. Document the experimental status prominently.

- **Prompt complexity.** Team orchestration prompts are significantly more complex than single-agent prompts. The team lead must correctly create tasks, assign agents, handle messages, and synthesize results. Mitigation: well-tested slash command prompts, clear task decomposition patterns.

- **Teammate coordination failures.** Agents may fail to mark tasks complete, send messages to wrong recipients, or go idle prematurely. Mitigation: the team lead monitors TaskList and can reassign or nudge stuck teammates.

### Low risk

- **No nested teams.** Teammates cannot spawn their own teams. This limits recursion depth but is fine for the two-to-three agent configurations we're targeting.

- **Session resumption.** `/resume` does not restore in-process teammates. If a session breaks mid-team, work must restart. Mitigation: team artifacts are committed incrementally, so partial progress is preserved in git.

## 6. Open Questions

1. **Model selection for read-only agents.** Should Training Partner and Skeptical Client agents use `haiku` for cost savings, or `inherit` (opus) for quality? The quality of adversarial critique may degrade with a smaller model. Needs experimentation.

2. **Team naming convention.** Should team names follow feature slugs (e.g., `push-hands-review-{slug}`) or use a fixed pattern? Affects `~/.claude/tasks/` directory structure.

3. **Artifact ownership.** When both a Proposer and Training Partner contribute to a PRP during `/review-plan-team`, who commits the final artifact? The team lead, or the Proposer teammate?

4. **Escalation from sequential to team.** Should users be able to start with `/review-plan` and escalate to team mode mid-review? Or is team mode a fresh invocation?

5. **Experimental feature gate.** Should we document the `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` setup requirement, or wait for GA? The feature may lose the experimental flag before this template ships widely.

## 7. Out of Scope

- **Replacing sequential commands.** Team mode augments, not replaces. `/review-plan`, `/execute-prp`, `/review-code`, and `/security-audit` remain unchanged.
- **Light tier support.** Team mode adds overhead inappropriate for lightweight changes. Light tier is excluded.
- **Automated tier escalation.** This PRD does not implement automatic switching between sequential and team modes based on change complexity.
- **Persistent teams.** Teams are ephemeral, created and destroyed within a single pipeline stage. Long-running teams that span multiple stages are not in scope.
- **Worktree integration.** The existing `worktree-setup.sh` parallel track system is orthogonal to agent teams. Integration between the two is a future concern.
- **CI/CD integration for team mode.** GitHub Actions workflows are not modified to verify team-mode artifacts differently from sequential-mode artifacts — the artifacts are identical in format and path.
- **Custom stance definitions by end users.** This PRD defines the five canonical stances as agent definitions. A future PRD could allow users to define additional stances.
