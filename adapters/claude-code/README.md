# Claude Code Adapter

Claude Code is the reference implementation for Push Hands. All methodology features are natively supported.

## Feature Matrix

| Feature | Support | Implementation |
|---------|---------|----------------|
| Slash commands | Native | `.claude/commands/*.md` |
| Agent stances | Native (tool-level enforcement) | `.claude/agents/*.md` with `tools:` frontmatter |
| Agent teams | Native (concurrent execution) | `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` |
| TDD enforcement | Via commands + hooks | `/execute-prp`, `/execute-team`, `push-hands-guard.sh` |
| Permission sandboxing | Native | `.claude/settings.local.json` |
| Project context | Native | `CLAUDE.md` (auto-loaded) |
| Methodology hooks | Native | `.claude/hooks/*.sh` |

## How It Works

### Slash Commands (`.claude/commands/`)

Each `.claude/commands/*.md` file defines a workflow stage as a self-contained prompt. When the user invokes `/command-name`, Claude Code loads the file content and executes it.

- 7 sequential commands: `new-feature`, `generate-prp`, `review-plan`, `execute-prp`, `review-code`, `security-audit`, `update-claude-md`
- 3 team commands: `review-plan-team`, `execute-team`, `security-audit-team`

**Pipeline change:** `/new-feature` now auto-generates both PRD and PRP in one invocation (standard/full tiers).

### Agent Definitions (`.claude/agents/`)

Each `.claude/agents/*.md` file defines a stance with YAML frontmatter specifying tool restrictions. The `tools:` field whitelists which tools the agent can use, enforcing stance constraints at the platform level.

| Stance | Tools | Enforcement |
|--------|-------|-------------|
| Proposer | Read, Write, Edit, Bash, Grep, Glob | Full access |
| Training Partner | Read, Grep, Glob, Bash | No Write/Edit — cannot modify files |
| Code Reviewer | Read, Grep, Glob, Bash | No Write/Edit — reviews only |
| Security Auditor | Read, Grep, Glob, Bash | No Write/Edit — Bash for PoC testing |
| Skeptical Client | Read, Grep, Glob | No Write/Edit/Bash — pure analysis |

All agent definitions use `model: opus` explicitly (not `inherit`).

### Team Mode

Agent teams run stances as concurrent teammates. Requires `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in Claude Code settings. See [docs/guides/agent-teams.md](../../docs/guides/agent-teams.md) for setup, known bugs, workarounds, and usage.

**Known limitations:**
- Agent definitions don't load for team members (bug #24316). Team commands inline stance definitions in spawn prompts as a workaround.
- Model inheritance is unreliable (bug #32368). Team commands and the `enforce-opus-teams.sh` hook require `model: "opus"` explicitly.
- Team commands include health checks and abort-and-defer: if teammates don't start within 90 seconds, the team shuts down and reports to the user. Never falls back to solo implementation.

### Methodology Hooks (`.claude/hooks/`)

Three hooks enforce the Push Hands methodology:

| Hook | Event | Purpose |
|------|-------|---------|
| `push-hands-guard.sh` | UserPromptSubmit | Injects methodology + TDD reminder before agent starts work |
| `block-solo-implementation.sh` | PreToolUse (Edit/Write) | Blocks direct src/tests edits on standard/full tier branches |
| `enforce-opus-teams.sh` | PreToolUse (Agent) | Requires `model: "opus"` for all team agent spawns |

## Relationship to Portable Templates

Claude Code commands are self-contained implementations of the methodology defined in `prompts/workflows/`. They add Claude Code-specific wiring:

- `$ARGUMENTS` for user input injection
- Tool names (`Read`, `Write`, `Edit`, `Bash`, `Grep`, `Glob`)
- Team primitives (`TeamCreate`, `SendMessage`, `TaskCreate`, etc.)
- Agent spawning with `subagent_type` references
- Inline stance definitions for team commands (workaround for bug #24316)

The portable templates in `prompts/` are the canonical methodology reference. Claude Code commands are the reference adapter.

### Sync Convention

When the methodology changes:
1. Update `prompts/workflows/` first (canonical source)
2. Update `.claude/commands/` to match
3. Run `scripts/check-prompt-sync.sh` to verify alignment

## Setup

`./scripts/setup.sh` handles everything — hook installation, directory creation, file verification, optional teammate wrapper setup. No additional steps needed for Claude Code.
