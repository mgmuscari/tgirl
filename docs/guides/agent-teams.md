# Agent Teams Mode

Team mode runs Push Hands stances as concurrent agent teammates rather than sequential invocations. Same artifact types at the same paths, with additional fields capturing the dialectical exchange.

## Prerequisites

Claude Code with agent teams enabled. Add to your Claude Code settings (`.claude/settings.json` or user-level settings):

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

**Note:** Agent teams is an experimental Claude Code feature. The API surface may change.

## Commands

| Command | Replaces | What It Does |
|---------|----------|--------------|
| `/review-plan-team <prp>` | `/review-plan` | Training Partner critiques PRP while Proposer revises in real time |
| `/execute-team <prp>` | `/execute-prp` + `/review-code` | Proposer implements tasks; Code Reviewer reviews each commit before the next begins |
| `/security-audit-team` | `/security-audit` | Security Auditor and Skeptical Client debate findings as separate agents |

All team commands produce artifacts at the **same paths** as their sequential equivalents, with additional team-specific fields (exchange trails, severity adjustments). Standard fields remain in the same position — team-mode artifacts are strict supersets. CI, close-feature.sh, and the PR template work without modification.

## When to Use Team Mode vs. Sequential

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Quick review of a small PRP | `/review-plan` | Lower cost, sufficient for simple plans |
| Complex PRP with many integration points | `/review-plan-team` | Real-time feedback catches issues before the PRP is finalized |
| Simple implementation (few tasks) | `/execute-prp` then `/review-code` | Sequential is cheaper |
| Large implementation (many tasks) | `/execute-team` | Incremental review catches issues per-commit, not post-hoc |
| Security audit (any complexity) | `/security-audit-team` | Dual-agent dialectic produces more defensible reports |

## Pipeline with Team Mode

Team commands slot into the same pipeline positions as their sequential equivalents. You can mix sequential and team commands freely:

```
/generate-prp → /review-plan-team → /execute-team → /security-audit-team
```

Or mix:

```
/generate-prp → /review-plan → /execute-team → /security-audit
```

**Important:** `/review-plan-team` requires an existing PRP. It enhances the review phase — it does not generate the PRP from scratch. Always run `/generate-prp` first.

## Cost Implications

Each teammate is a separate Claude instance. A team with 2 agents costs approximately 2-3x a single invocation. Team mode is opt-in precisely because of this cost.

**Tips for reducing cost:**
- Use team mode only when the added dialectical quality justifies the cost
- For cost-sensitive projects, edit agent definitions in `.claude/agents/` to set `model: haiku` for critic agents (training-partner, code-reviewer, skeptical-client). This reduces cost but may degrade critique quality.
- `/security-audit-team` is the highest-value team command — the dual-agent dialectic materially improves audit defensibility

## Customizing Agent Definitions

Agent definitions live in `.claude/agents/`. Each file has YAML frontmatter (name, tools, model) and a markdown body (system prompt). You can customize:

- **Tools:** Change which tools an agent can access
- **Model:** Use `model: opus` (required for team mode). Do NOT use `model: inherit` — it doesn't resolve properly (bug #32368).
- **Prompt body:** Adjust the agent's behavior, communication style, or review focus

See `AGENTS.md` for the mapping between stances and agent definitions.

## Known Bugs and Workarounds

### Bug #24316: Agent definitions don't load for team members (OPEN)

**Problem:** When `team_name` is set on an Agent spawn, `subagent_type` is ignored. All teammates spawn as `general-purpose` with full tool access. Tool restrictions from `.claude/agents/*.md` are NOT enforced.

**Impact:** Critic agents (training partner, code reviewer, skeptical client) get Write/Edit tools they shouldn't have. Stance priming from agent definitions is lost.

**Workaround:** Team commands inline the full stance definition — character, constraints, and tool restrictions — directly in the spawn prompt. This is redundant with agent definitions but necessary. The inline prompts explicitly tell critic agents "You CANNOT and MUST NOT modify files. You have no Write or Edit tools."

**Status:** Open, 26+ comments. No ETA for fix.

### Bug #32368: Model inheritance broken

**Problem:** Teammates get hardcoded model IDs instead of inheriting parent config. `model: inherit` doesn't resolve properly.

**Impact:** Agents may spawn with wrong model, go idle, or produce poor results.

**Workaround:**
1. All agent definitions use `model: opus` explicitly (not `inherit`)
2. All team spawn calls include `model: "opus"` explicitly
3. The `enforce-opus-teams.sh` hook blocks team spawns without `model: "opus"`
4. Optional: Set `CLAUDE_CODE_TEAMMATE_COMMAND` to `./scripts/claude-teammate-wrapper.sh`

**Status:** Partial fix in v2.1.44-45, still broken for many configurations.

### Bug: Environment propagation

**Problem:** Bedrock/proxy env vars don't propagate to tmux-spawned teammates.

**Workaround:** Set environment variables in `.claude/settings.json` `env` block instead of shell environment.

### Health check and abort mechanism

All team commands include a health check after spawning:
- If teammates haven't started within 90 seconds, the team is shut down
- The user is notified with options: retry the team command, use sequential fallback, or check Claude Code version
- Teams **never** silently fall back to solo implementation

## Limitations

- **Experimental:** Feature-gated, may change. Requires `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`.
- **No session resumption:** If a session breaks mid-team, work must restart. Partial progress is preserved in git (commits are incremental).
- **One team per session:** Cannot run multiple team commands simultaneously.
- **File conflict prevention:** Critic agents are read-only (no Write/Edit tools). In `/execute-team`, the Proposer and Reviewer coordinate via message-gating to avoid concurrent git operations.
- **Agent definitions not loaded:** See bug #24316 above. Stance enforcement relies on inline prompts in team commands.

## How It Works

### Message-Based Coordination

Unlike the sequential pipeline where each stage runs to completion before the next begins, team mode agents communicate directly via `SendMessage`. The Training Partner sends yield points to the Proposer as they're found. The Proposer responds and revises. The exchange continues until it stabilizes.

### Tool-Level Stance Enforcement

In sequential mode, stance constraints are enforced by prompt instructions ("you cannot write code"). In team mode, constraints are enforced at the tool level — the Training Partner agent literally does not have the `Write` or `Edit` tools. This makes the constraint architectural, not behavioral.

### The Lead Writes Artifacts

Review and audit artifacts are written by the team lead (the main Claude Code session), not by individual agents. Agents communicate findings via messages; the lead synthesizes them into the standard artifact format. This keeps critic agents truly read-only while producing artifacts identical to those from sequential commands.
